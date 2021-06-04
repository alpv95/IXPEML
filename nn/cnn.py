import os
import h5py
import math
import random
import inspect
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as weight_init
from torch.autograd import Variable
from scipy.special import expit
from nn import cnn_loss
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import apex
import shutil
import gc

def softmax(x, axis):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=axis)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, p_drop=0.3):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.drop = nn.Dropout(p=p_drop)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.drop(F.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, outputtype, p_drop, input_channels):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.outputtype = outputtype
        self.p_drop = p_drop
        self.input_channels = input_channels
        
        if outputtype == '1ang' or outputtype == '1energy':
            output_num = 1
        elif outputtype == '2pos' or outputtype == 'abs_pts':
            output_num = 2
        elif outputtype == '3pos':
            output_num = 3
        elif outputtype == '7pos2err':
            output_num = 9
        elif outputtype == '2pos1err':
            output_num = 3
        elif outputtype == '5pos1err':
            output_num = 6
        elif outputtype == '5pos4err':
            output_num = 9

        self.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, output_num)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, p_drop=self.p_drop))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



def TrackNet(outputtype): 
    return ResNet(BasicBlock, [1,1,1,1], outputtype, 0, 2)


class TrackAngleRegressor:
    """Interface for convolutional net for calculating track angle"""
    def __init__(self, losstype, load_checkpoint=None, input_channels=2, pixels=50, use_gpu=True, **kwargs):
        self.pixels = pixels
        self.input_channels = input_channels
        self.losstype = losstype

        # Set GPU use if desired.
        if use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            print("NN on GPU \n")
            gc.collect()
            torch.cuda.empty_cache()
        else:
            use_gpu = False
            device = torch.device("cpu")
            print("NN on CPU")

        loss_to_output = {'mserrall2': '5pos1err', 'mserrall3':'5pos4err', 
                            'energy':'1energy', 'tailvpeak':'1ang', 
                            'tailvpeak2':'3pos',}
        self.outputtype = loss_to_output[self.losstype]

        # Build net
        self.net = TrackNet(self.outputtype)
        self.net.to(device)

        if load_checkpoint is not None:
            print("=> loading checkpoint '{}'".format(load_checkpoint))
            self.net.load_state_dict(torch.load(load_checkpoint))
            

    def train(self, train_loader, val_loader, train_sampler, checkpoint_dir, local_rank=0, world_size=1, 
                n_epochs=10, batch_size=256, optim_method='mom', rng_seed=None, **kwargs):
        """Train model on provided data. Uses Adam optimizer.

        Args:
            X: n_samples x h x w float32 ndarray of track images. Can also accept n_samples x 1 x h x w with the
                dummy channel dim already present.
            y: n_samples float32 ndarray of angles
            losstype: 'mse', 'ang', 'cos', str for loss function type.
            lr: float, learning rate
            wd: float, weight decay
            n_epochs: int, number of epochs (runs thru whole batch) to train
            batch_size: int, number of samples per batch
            rng_seed: None or int, random number seed used for all seeds to get consistent results
            **kwargs: additional hparams for the net. See TrackNet's constructor's hparams dict.

        Returns: if track_stats=False, returns mse; else returns tuple of (mse, stats)
            mse: float, mean squared error in angular distance between predicted and y
            stats: dict of stats taken at steps
        """
        # Set RNG seeds
        # if rng_seed is not None:
        #     if verbose > 0:
        #         print('Setting RNG seed to {}'.format(rng_seed))
        #     random.seed(rng_seed)
        #     np.random.seed(rng_seed)
        #     torch.manual_seed(rng_seed)

        # Ensure X has proper dims
        n_samples = len(train_loader.dataset)
        hparams = kwargs

        # Set loss function and make required changes
        if self.losstype == 'mserrall2':
            criterion = cnn_loss.MSErrLossAll2(alpha=hparams['alpha_loss'], 
                            lambda_abs=hparams['lambda_abs'], lambda_E=hparams['lambda_E']).cuda()
            val_criterion = cnn_loss.MSErrLossAll2(size_average=False, alpha=hparams['alpha_loss'], 
                            lambda_abs=hparams['lambda_abs'], lambda_E=hparams['lambda_E']).cuda()
            label_number = 5
        elif self.losstype == 'mserrall3':
            criterion = cnn_loss.MSErrLossAll3().cuda()
            val_criterion = cnn_loss.MSErrLossAll3(size_average=False,).cuda()
            label_number = 5
        elif self.losstype == 'energy':
            criterion = cnn_loss.MSErrLoss().cuda()
            val_criterion = cnn_loss.MSErrLoss(size_average=False).cuda()
            label_number = 1
        elif self.losstype == 'tailvpeak':
            criterion = cnn_loss.BinaryLoss().cuda()
            val_criterion = cnn_loss.BinaryLoss(size_average=False).cuda()
            label_number = 1
        elif self.losstype == 'tailvpeak2':
            criterion = nn.CrossEntropyLoss().cuda()
            val_criterion = nn.CrossEntropyLoss(size_average=False).cuda()
            label_number = 1
        else:
            raise ValueError('Loss type not recognized')

        # Build optimization method
        if optim_method == 'RLRP': #Cyclic learning rate with decaying amplitude
            optimizer = optim.SGD(self.net.parameters(), lr=hparams['lr'], momentum=0.9, weight_decay=hparams['wd'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=9, min_lr=0.00001)
        elif optim_method == 'mom': #Normal SGD momentum schedule but with initial rise to prevent loss divergence
            optimizer = optim.SGD(self.net.parameters(), lr=hparams['lr'], momentum=0.9, weight_decay=hparams['wd'])
            def lr_lambda(ep):
                if ep <= 10:
                    L = 0.3 * ep
                elif ep > 10 and ep <= 40:
                    L = 1
                elif ep > 40 and ep <= 90:
                    L = 0.1
                elif ep > 90 and ep <= 130:
                    L = 0.01
                else:
                    L = 0.001               
                return L
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda,)
        else:
            raise("ERROR: Optimizer not understood!")

        # Calculate n_steps from n_epochs
        n_steps = n_epochs * int(math.ceil(n_samples / batch_size))
        # Added after model and optimizer construction
        self.net = apex.parallel.convert_syncbn_model(self.net)
        self.net, optimizer = amp.initialize(self.net, optimizer, opt_level=hparams['opt_level'])
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.net = DDP(self.net, delay_allreduce=True)
        
        best_val_loss = 100
        print('Training for {} steps'.format(n_steps))
        steps = []
        losses = []
        val_steps = []
        val_losses = []

        #Training loop
        for epoch in range(1,n_epochs + 1):
            if torch.cuda.device_count() > 1:
                train_sampler.set_epoch(epoch)

            self.net.train()
            cum_loss = 0
            cum_batch = 0
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                #reshape X_batch and y_batch here from augment
                X_batch = X_batch.view([-1, self.input_channels, self.pixels, self.pixels])
                y_batch = y_batch.view([-1,label_number])
                if label_number == 1:
                   y_batch = y_batch.squeeze()

                X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
                y_hat_batch = self.net(X_batch)
                loss = criterion(y_hat_batch, y_batch)
                optimizer.zero_grad()
                # loss.backward() changed to:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()
                
                if batch_idx % 150 == 0:                        
                    if torch.cuda.device_count() > 1:
                        losses.append(reduce_tensor(loss.data, world_size).item())
                        cum_loss += reduce_tensor(loss.data, world_size).item(); cum_batch += 1
                    else:
                        losses.append(loss.item())
                        cum_loss += loss.item(); cum_batch += 1
                    steps.append(epoch * math.ceil(len(train_loader.dataset) / batch_size) + batch_idx)
                    torch.cuda.synchronize()
                    
                    if local_rank == 0:    
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * batch_size, len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), cum_loss / cum_batch))
 

            #evaluate on validation set
            if ((epoch-1) % 2 == 0):
                self.net.eval()
                val_loss = 0
                val_acc = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.view([-1, self.input_channels, self.pixels, self.pixels])
                        y_batch = y_batch.view([-1,label_number]) 
                        if label_number == 1:
                            y_batch = y_batch.squeeze()

                        X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
                        y_hat_batch = self.net(X_batch)
                        if torch.cuda.device_count() > 1:
                            val_loss += reduce_tensor(val_criterion(y_hat_batch, y_batch).data).item()
                        else:
                            val_loss += val_criterion(y_hat_batch, y_batch).item() # sum up batch loss

                        if self.losstype == 'tailvpeak':
                            val_acc += reduce_tensor(torch.eq(torch.gt(y_hat_batch.squeeze(),0), y_batch).sum().float().data).item()
                        elif self.losstype == 'tailvpeak2':
                            val_acc += np.sum(np.argmax(y_hat_batch.data.cpu().numpy(),axis=-1) == y_batch.data.cpu().numpy() )

                val_loss /= len(val_loader.dataset)
                val_losses.append(val_loss)
                val_steps.append(epoch * math.ceil(n_samples / batch_size) )

                if local_rank == 0:
                    if (self.losstype == 'tailvpeak' or self.losstype == 'tailvpeak2'):
                        val_acc /= len(val_loader.dataset)
                        print('\n Validation Set Accuracy: {:.4f}\n'.format(val_acc))
                    print('\nValidation set - Average loss: {:.4f}\n'.format(val_loss))

            if (epoch % 10 == 0) and epoch >= 25 and local_rank == 0:
                print('\n Saving checkpoint ' + "net_" + str(epoch) + ".ptmodel\n")
                is_best = val_loss < best_val_loss
                best_val_loss = min(val_loss, best_val_loss)
                self.dump(checkpoint_dir + "/" + optim_method + "_" + str(batch_size) + "_" + str(epoch) + ".ptmodel", is_best) 

            if np.isnan(cum_loss): #check for loss divergence, don't waste gpu time
                break
            if optim_method == "RLRP":
                scheduler.step(val_loss)
            else:
                scheduler.step() #checking for lr decay
            
        print('done training.')

        results = self._eval(train_loader, use_gpu=use_gpu) 

        return results

    def test(self, data_loader, y_exists=True, use_gpu=True, output_all=False, output_vals=()):
        """Run inference on provided data and compare to expected results. Supports a variety of outputs."""
        return self._eval(data_loader, y_exists, use_gpu=use_gpu, output_all=output_all, output_vals=output_vals)

    def predict(self, data_loader, use_gpu=True, one_batch=False, output_vals=()):
        """Run inference on provided data, outputting just the predicted angles"""
        metrics = self._eval(data_loader, y_exists=False, use_gpu=use_gpu, one_batch=False, output_all=True, output_vals=output_vals)
        return metrics['y_hat_angles']

    def dump(self, model_file, is_best):
        if is_best:
            torch.save(self.net.module.state_dict(), model_file + "B")
        else:
            torch.save(self.net.module.state_dict(), model_file)


    def _drop_eval(self,m):
        if type(m) == nn.Dropout:
            m.train()

    def _eval(self, data_loader, y_exists=True, use_gpu=True, one_batch=False, output_all=False, output_vals=()):
        """Evaluate performance metrics. Splits into batches to evaluate. Optionally outputs additional metrics.

        Args:
            X: same as in train
            y: same as in train, except can also be None
            use_gpu: same as train
            output_all: bool, whether to return a tuple, w/ the 2nd element as extended outputs
            output_vals: tuple of strs, names of self.net's forward_all() method outputs to save

        Returns: If output_all=False, returns just MSE; else returns tuple of (mse, metrics)
            mse: float, mean squared angle error
            metrics: dict of metrics
        """
        output_type = self.net.outputtype
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.net = nn.DataParallel(self.net)

        self.net.eval()

        if use_gpu and torch.cuda.is_available():
           device = torch.device("cuda")
           torch.cuda.empty_cache()
           print("Evaluating on GPU \n")
           self.net.cuda()
        else:
           use_gpu = False
           device = torch.device("cpu")
           print("Evaluating on CPU \n")
           
        n_samples = len(data_loader.dataset)
        y_hats = np.array([],dtype=np.float32)

        with torch.no_grad():
            for batch_idx, X_batch in enumerate(data_loader):
                X_batch = X_batch.view([-1, self.input_channels, self.pixels, self.pixels])

                X_batch = X_batch.to(device)
                y_hat_batch = self.net(X_batch).cpu().data.numpy()
                y_hats = np.append(y_hats, y_hat_batch)

                if batch_idx % 100 == 0:
                    print("{} of {} reconstructed.".format(int(batch_idx * X_batch.shape[0] / 6), n_samples))

 
        if output_type == '1ang':
            #Apply sigmoid to tailvpeak outputs
            y_hat_angles = expit(y_hats)

        elif output_type == '1energy':
            y_hat_angles = y_hats

        elif output_type == '3pos':
            #Apply sigmoid to tailvpeak outputs
            y_hat_angles = softmax(y_hats, axis=-1)

        elif output_type == '5pos1err':
            y_hats = y_hats.reshape(-1,6)
            y_hat_angles = np.array(( np.arctan2(y_hats[:,1], y_hats[:,0]), np.sqrt(np.exp(y_hats[:,2])), y_hats[:,3],
                                        y_hats[:,4], y_hats[:,5]))

        elif output_type == '5pos4err':#dont worry about error for now
            y_hats = y_hats.reshape(-1,9)
            y_hat_angles = np.array(( np.arctan2(y_hats[:,1], y_hats[:,0]), np.sqrt(np.exp(y_hats[:,2])), y_hats[:,3],
                                        y_hats[:,4], y_hats[:,5], y_hats[:,6], y_hats[:,7], y_hats[:,8]))

        metrics = {
            'y_hat': y_hats,  # the actual predictions - could be a vec or a 2 col mat
            'y_hat_angles': y_hat_angles  # predicted angles
        }

        # Combine layer results and store in metrics dict
        if len(output_vals) > 0:
            for val in output_vals:
                metrics[val] = np.concatenate(x_alls[val], axis=0)

        return metrics

def reduce_tensor(tensor, world_size=1):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt
