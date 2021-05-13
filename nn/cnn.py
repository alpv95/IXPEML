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
        elif outputtype == 'CE':
            output_num = 6

        self.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, output_num)
        # self.sigmoid = nn.Sigmoid() #for tailvpeak training only, now included separately in Binary loss function

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
        # out = self.sigmoid(out) #for tailvpeak training only
        return out



def TrackNet(pixels, hparams=None, params=None): 
    hparams_ = {
        'outputtype': '1ang',  # 1 angle in [-pi,pi)
        'pixels': pixels,
    }

    if hparams is None:
        hparams = hparams_
    else:
        hparams = {**hparams_, **hparams}

    return ResNet(BasicBlock, [1,1,1,1], hparams['outputtype'], hparams['dropout'], hparams['input_channels'])


class TrackAngleRegressor:
    """Interface for convolutional net for calculating track angle"""
    def __init__(self, load_checkpoint=None, input_channels=1, pixels=50):
        self.net = None
        self.opts = {}
        self.pixels = pixels
        self.input_channels = input_channels
        if load_checkpoint is not None:
            #pixels = model['hparams']['pixels']
            #self.net = TrackNet(pixels, hparams=model['hparams'], params=model['params'])
            if torch.cuda.is_available():
               self.net = torch.load(load_checkpoint)
               if torch.cuda.device_count() > 1:
                   print("Let's use", torch.cuda.device_count(), "GPUs!")
                   self.net = nn.DataParallel(self.net)
            else:
               self.net = torch.load(load_checkpoint, map_location='cpu')
            #self.opts = model['opts']
            

    def train(self, train_loader, val_loader, checkpoint_dir, meas_loader=None,
              losstype='mse', lr=1e-1, wd=1e-4, n_epochs=10, batch_size=128, optim_method='mom',
              verbose=1, rng_seed=None, use_gpu=True, track_stats=False,
              **kwargs):
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
            verbose: 0 or 1, amount of output to print
            rng_seed: None or int, random number seed used for all seeds to get consistent results
            use_gpu: bool, whether to use the (first) GPU. Only has effect if a GPU is available.
            track_stats: bool, whether to track stats throughout training. If true, returns a tuple of mse and stats.
            **kwargs: additional hparams for the net. See TrackNet's constructor's hparams dict.

        Returns: if track_stats=False, returns mse; else returns tuple of (mse, stats)
            mse: float, mean squared error in angular distance between predicted and y
            stats: dict of stats taken at steps
        """
        # Save keyword opts
        args = locals()
        for param in inspect.signature(self.train).parameters.values():
            if param.default is not param.empty:
                self.opts[param.name] = args[param.name]

        # Set RNG seeds
        if rng_seed is not None:
            if verbose > 0:
                print('Setting RNG seed to {}'.format(rng_seed))
            random.seed(rng_seed)
            np.random.seed(rng_seed)
            torch.manual_seed(rng_seed)

        # Set GPU use if desired. Only training uses the GPU.
        if use_gpu and torch.cuda.is_available():
            if rng_seed is not None:
                torch.cuda.manual_seed(rng_seed)
            device = torch.device("cuda")
            print("Training on GPU \n")
        else:
            use_gpu = False
            device = torch.device("cpu")
            print("Training on CPU")

        # Ensure X has proper dims
        n_samples = len(train_loader.dataset)
        pixels = train_loader.dataset.pixels
        hparams = kwargs
        self.input_channels = hparams['input_channels']

        # Set loss function and make required changes
        if losstype == 'mserrall2':
            criterion = cnn_loss.MSErrLossAll2(alpha=hparams['alpha_loss'], 
                            lambda_abs=hparams['lambda_abs'], lambda_E=hparams['lambda_E'])
            val_criterion = cnn_loss.MSErrLossAll2(size_average=False, alpha=hparams['alpha_loss'], 
                            lambda_abs=hparams['lambda_abs'], lambda_E=hparams['lambda_E'])
            label_number = 5
            hparams['outputtype'] = '5pos1err'
        elif losstype == 'mserrall3':
            criterion = cnn_loss.MSErrLossAll3()
            val_criterion = cnn_loss.MSErrLossAll3(size_average=False,)
            label_number = 5
            hparams['outputtype'] = '5pos4err'
        elif losstype == 'energy':
            criterion = cnn_loss.MSErrLoss()
            val_criterion = cnn_loss.MSErrLoss(size_average=False)
            label_number = 1
            hparams['outputtype'] = '1energy'
        elif losstype == 'tailvpeak':
            criterion = cnn_loss.BinaryLoss()
            val_criterion = cnn_loss.BinaryLoss(size_average=False)
            label_number = 1
            hparams['outputtype'] = '1ang'
        elif losstype == 'tailvpeak2':
            criterion = nn.CrossEntropyLoss()
            val_criterion = nn.CrossEntropyLoss(size_average=False)
            label_number = 1
            hparams['outputtype'] = '3pos'
        else:
            raise ValueError('Loss type not recognized')

        # Build net
        self.net = TrackNet(pixels, hparams=hparams)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.net = nn.DataParallel(self.net)
        self.net.to(device)

        if verbose > 0:
            print('Built net')
            print(self.net)

        # Build optimization method
        if optim_method == 'RLRP': #Cyclic learning rate with decaying amplitude
            optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=11, min_lr=0.00001)
        elif optim_method == 'mom': #Normal SGD momentum schedule but with initial rise to prevent loss divergence
            optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
            def lr_lambda(ep):
                if ep <= 10:
                    L = 0.1 * ep
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
        

        if verbose:
           print('Training for {} steps'.format(n_steps))
        if track_stats:
           steps = []
           losses = []
           val_steps = []
           val_losses = []
           mod_factor = [10]
           phi_0s = [10]
        #ALP New and improved train
        for epoch in range(1,n_epochs + 1):
            self.net.train()
            cum_loss = 0
            cum_batch = 0
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                #reshape X_batch and y_batch here from augment
                X_batch = X_batch.view([-1, self.input_channels, self.pixels, self.pixels])
                y_batch = y_batch.view([-1,label_number])
                if label_number == 1:
                   y_batch = y_batch.squeeze()

                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                y_hat_batch = self.net(X_batch)
                loss = criterion(y_hat_batch, y_batch)
                loss.backward()
                optimizer.step()
                if (verbose > 0 or track_stats) and batch_idx % 100 == 0:
                   losses.append(loss.item())
                   cum_loss += loss.item(); cum_batch += 1
                   steps.append(epoch * math.ceil(len(train_loader.dataset) / batch_size) + batch_idx)
                   print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                   epoch, batch_idx * batch_size, len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), cum_loss / cum_batch))
 

            #evaluate on validation set
            if ((epoch-1) % 3 == 0):
               self.net.eval()
               val_loss = 0
               val_acc = 0
               with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.view([-1, self.input_channels, self.pixels, self.pixels])
                        y_batch = y_batch.view([-1,label_number]) 
                        if label_number == 1:
                           y_batch = y_batch.squeeze()

                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                        y_hat_batch = self.net(X_batch)
                        val_loss += val_criterion(y_hat_batch, y_batch).item() # sum up batch loss
                        if losstype == 'CE':
                            val_acc += np.sum((np.argmax(y_hat_batch.data.cpu().numpy(), axis=1) == y_batch.data.cpu().numpy())*1)
                        elif losstype == 'tailvpeak':
                            val_acc += np.sum((np.where(np.squeeze(expit(y_hat_batch.data.cpu().numpy())) > 0.5, 1, 0) == y_batch.data.cpu().numpy())*1)
                        elif losstype == 'tailvpeak2':
                            val_acc += np.sum(np.argmax(y_hat_batch.data.cpu().numpy(),axis=-1) == y_batch.data.cpu().numpy() )
               val_loss /= len(val_loader.dataset) #(y_batch.shape[0] / batch_size)
               val_losses.append(val_loss)
               val_steps.append(epoch * math.ceil(n_samples / batch_size) )
               if losstype == 'CE' or losstype == 'tailvpeak' or losstype == 'tailvpeak2':
                  val_acc /= len(val_loader.dataset)
                  print('\n Validation Set Accuracy: {:.4f}\n'.format(val_acc))
               print('\nValidation set - Average loss: {:.4f}\n'.format(val_loss))

            if ((epoch-1) % 10 == 0 and any(losstype == L for L in ['cos','mserr','maerr','mserrall1','mserrall2']) and meas_loader):
               self.net.eval()
               mod = 0; phi0 = 0
               C1 = 0; C2 = 0
               with torch.no_grad():
                    y_hats = self.predict(meas_loader)

               N_elliptic = len(y_hats[0,:])
               Q = sum(np.cos(2*y_hats[0,:])) / N_elliptic
               U = sum(np.sin(2*y_hats[0,:])) / N_elliptic
               
               mod = 2*np.sqrt(Q**2 + U**2)
               phi0 = np.arctan2(U,Q)/2 
               
               print("y_hat examples: ", y_hats[:20])
               mod_factor.append(mod)
               phi_0s.append(phi0)
               print('\nMeasured set - Modulation Factor & Phi_0: {:.4f}, {:.4f}\n'.format(mod, phi0))

            if ( ((epoch-1) % 10 == 0 and epoch >= 10) ):
                print('\n Saving checkpoint ' + "net_" + str(epoch) + ".ptmodel\n")
                self.dump(checkpoint_dir + "/" + optim_method + "_" + str(batch_size) + "_" + str(epoch) + ".ptmodel") 

            if np.isnan(cum_loss): #check for loss divergence, don't waste gpu time
                break
            if optim_method == "RLRP":
                scheduler.step(val_loss)
            else:
                scheduler.step() #checking for lr decay
            
        if verbose > 0:
            print('done training.')

        results = self._eval(train_loader, use_gpu=use_gpu)  # feed in 4d X and 2d y for losstype=cos, 1d y for losstype!=cos

        if track_stats:
            stats = {
                'step': steps,
                'loss': losses,
                'val_step': val_steps,
                'val_loss': val_losses,
                'mod_factor': mod_factor,
                'phi0': phi_0s,
            }
            return results, stats
        else:
            return results

    def test(self, data_loader, y_exists=True, use_gpu=True, output_all=False, output_vals=()):
        """Run inference on provided data and compare to expected results. Supports a variety of outputs."""
        return self._eval(data_loader, y_exists, use_gpu=use_gpu, output_all=output_all, output_vals=output_vals)

    def predict(self, data_loader, use_gpu=True, one_batch=False, output_vals=()):
        """Run inference on provided data, outputting just the predicted angles"""
        metrics = self._eval(data_loader, y_exists=False, use_gpu=use_gpu, one_batch=False, output_all=True, output_vals=output_vals)
        return metrics['y_hat_angles']
    
    def save_checkpoint(self, epoch):
        """Save net during training"""
        net = self.net.cpu()
        torch.save(
            net.state_dict(),
            str(net.__class__.__name__) + "_" + str(epoch) + ".ptmodel",
        )

    def load(self, filename):
        self.net.load_state_dict(torch.load(filename))

    def dump(self, model_file):
        if torch.cuda.device_count() > 1:
            torch.save(self.net.module, model_file)
        else:
            torch.save(self.net, model_file)

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
        if torch.cuda.device_count() > 1:
            output_type = self.net.module.outputtype  # different behavior depending on net output type
        else:
            output_type = self.net.outputtype

        self.net.eval()

        if use_gpu and torch.cuda.is_available():
           device = torch.device("cuda")
           print("Evaluating on GPU \n")
        else:
           use_gpu = False
           device = torch.device("cpu")
           print("Evaluating on CPU \n")

        n_samples = len(data_loader.dataset)
        y_hats = np.array([],dtype=np.float32)
            
        if output_all and len(output_vals) > 0:
            x_alls = {}
            for val in output_vals:
                x_alls[val] = []

        with torch.no_grad():
            for batch_idx, X_batch in enumerate(data_loader):
                X_batch = X_batch.view([-1, self.input_channels, self.pixels, self.pixels])

                X_batch = X_batch.to(device)
                y_hat_batch = self.net(X_batch).cpu().data.numpy()
                y_hats = np.append(y_hats, y_hat_batch)

                if output_all and len(output_vals) > 0:
                    x_all = self.net.forward_all(X_batch)
                    for val in output_vals:
                        x_alls[val].append(x_all[val].cpu().data.numpy())

 
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
