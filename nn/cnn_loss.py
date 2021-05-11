'''
Convolutional neural net image processing for inner/individual track angle prediction
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from scipy.special import i0,i1

class BinaryLoss(nn.Module):
    """
    Binary cross entropy loss for predicting tail vs. peak track probability.
    """
    def __init__(self, size_average=True, reduce=True,):
        super(BinaryLoss, self).__init__()
        self.bceloss = nn.BCELoss(size_average=size_average, reduce=reduce)
        self.sig = nn.Sigmoid()

    def forward(self, input, target):
        return self.bceloss(self.sig(input)[:,0], target,)

class MSErrLoss(nn.Module):
    """
    Simple MSE Loss for training to predict event energy only.
    """
    def __init__(self, size_average=True, reduce=True,):
        super(MSErrLoss, self).__init__()
        self.mseloss = nn.MSELoss(size_average=size_average, reduce=reduce)

    def forward(self, input, target):
        return self.mseloss(input[:,0], target,)


class I0Function(torch.autograd.Function):
    """
    Modified Bessel function I0 gradient for backpropagation.
    """

    @staticmethod
    def forward(ctx, input):
        #preprocess
        input_cpu = input.data.cpu().numpy()
        #full loss
        output = i0(input_cpu)
        grad = i1(input_cpu)
        ctx.save_for_backward(torch.Tensor(grad).to(input.device)) #caching these for backwards pass
        return torch.Tensor(output).to(input.device)

    @staticmethod
    def backward(ctx, grad_output):
        grad = ctx.saved_tensors[-1]

        return grad_output * grad

class I0(nn.Module):
    def __init__(self,):
        super(I0, self).__init__()

    def forward(self, input):
        return I0Function.apply(input)


class MSErrLossAll2(nn.Module):
    """ 
    Full NN loss function incorporating PE angular loss for both pi and 2pi, abs_pts xy loss, and MSE loss for energy.
    alpha parameter controls ratio between pi and 2pi PE angle vector loss.
    """
    besseli0 = I0()

    def __init__(self, size_average=True, reduce=True, alpha=1, lambda_abs=1, lambda_E=1):
        super(MSErrLossAll2, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.alpha = alpha
        self.lambda_abs = lambda_abs
        self.lambda_E = lambda_E

    def forward(self, input, target):
        norm = input[:,:2] / torch.sqrt(input[:,0]**2 + input[:,1]**2).unsqueeze(1)
        abs_pt = target[:,2:4]
        energy = target[:,4]
        loss = 0.5 * ((norm - target[:,:2])**2).sum(1) * self.alpha \
              - torch.exp(-input[:,2]) * ( (norm[:,0]**2 - norm[:,1]**2) * (target[:,0]**2 - target[:,1]**2) + 2*norm[:,0]*norm[:,1] * 2*target[:,0]*target[:,1] ) \
              + self.lambda_abs * ((input[:,3:5] - abs_pt)**2).sum(1) + self.lambda_E *  (input[:,5] - energy)**2 \
              + torch.log(self.besseli0(torch.exp(-input[:,2])))

        if self.reduce:
            if self.size_average:
                loss = torch.mean(loss)
            else:
                loss = torch.sum(loss)

        return loss


class MSErrLossAll3(nn.Module):
    """ 
    Full NN loss function incorporating PE angular loss for both pi and 2pi, abs_pts xy loss, and MSE loss for energy.
    alpha parameter controls ratio between pi and 2pi PE angle vector loss.
    """
    besseli0 = I0()

    def __init__(self, size_average=True, reduce=True,):
        super(MSErrLossAll2, self).__init__()
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, input, target):
        norm = input[:,:2] / torch.sqrt(input[:,0]**2 + input[:,1]**2).unsqueeze(1)
        abs_pt = target[:,2:4]
        energy = target[:,4]
        loss = - torch.exp(-input[:,6]) * (norm[:,0]*target[:,0] + norm[:,1]*target[:,1]) \
              - torch.exp(-input[:,2]) * ( (norm[:,0]**2 - norm[:,1]**2) * (target[:,0]**2 - target[:,1]**2) + 2*norm[:,0]*norm[:,1] * 2*target[:,0]*target[:,1] ) \
              + 0.5 * torch.exp(-input[:,7]) * ((input[:,3:5] - abs_pt)**2).sum(1) + 0.5 * torch.exp(-input[:,8]) *  (input[:,5] - energy)**2 \
              + torch.log(self.besseli0(torch.exp(-input[:,2])) * self.besseli0(torch.exp(-input[:,6])) ) + 0.5 * input[:,7] + 0.5 * input[:,8]

        if self.reduce:
            if self.size_average:
                loss = torch.mean(loss)
            else:
                loss = torch.sum(loss)

        return loss
