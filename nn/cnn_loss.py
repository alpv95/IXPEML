# Convolutional neural net image processing for inner/individual track angle prediction
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import scipy
from scipy.special import i0,i1

class BinaryLoss(nn.Module):
    def __init__(self, size_average=True, reduce=True,):
        super(BinaryLoss, self).__init__()
        self.bceloss = nn.BCELoss(size_average=size_average, reduce=reduce)
        self.sig = nn.Sigmoid()

    def forward(self, input, target):
        return self.bceloss(self.sig(input)[:,0], target,)

class MSErrLoss(nn.Module):
    def __init__(self, size_average=True, reduce=True,):
        super(MSErrLoss, self).__init__()
        self.mseloss = nn.MSELoss(size_average=size_average, reduce=reduce)

    def forward(self, input, target):
        return self.mseloss(input[:,0], target,)


class MSErrLossAll1(nn.Module):
    """Angle between input and target vectors, where input vector is an x,y point on the unit circle corresponding
    to input angle, and target vector has x,y points in [-1,1] x [-1,1] (L_Inf ball).

    Only the angle is used for optimization (for now), but the magnitude of the target vector also conveys some
    information - the certainty of the prediction/eccentricity of the system. Can report this (or a normalized
    version that accounts for the max length that's different from the axes vs the corners due to anisotropy).
    """
    def __init__(self, size_average=True, reduce=True, alpha=1, Z=None, lambda_abs=1, lambda_E=1):
        super(MSErrLossAll1, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.alpha = alpha
        self.Z = Z
        self.lambda_abs = lambda_abs
        self.lambda_E = lambda_E

    def forward(self, input, target):
        # Works when target is broadcastable
        norm = input[:,:2] / torch.sqrt(input[:,0]**2 + input[:,1]**2).unsqueeze(1)
        abs_pt = target[:,2:4]
        energy = target[:,4]
        loss = 0.5 * torch.exp(-input[:,2]) * ( ((norm - target[:,:2])**2).sum(1) * self.alpha #) + 0.5 * input[:,2]  
              + ( ( (norm[:,0]**2 - norm[:,1]**2) - (target[:,0]**2 - target[:,1]**2) )**2 + (2*norm[:,0]*norm[:,1] - 2*target[:,0]*target[:,1])**2 ) ) \
              + self.lambda_abs * ((input[:,3:5] - abs_pt)**2).sum(1) + 0.4*self.lambda_E * F.smooth_l1_loss(2.5*input[:,5],2.5*energy,reduction='none') \
              + 0.5 * input[:,2]

        if self.reduce:
            if self.size_average:
                loss = torch.mean(loss)
            else:
                loss = torch.sum(loss)
        if self.Z is not None:
            return loss + self.Z * self.Z_m(norm)
        else:
            return loss

    def Z_m(self, cs):
        n_modes = 5
        Z_m = ( ( self.Cnx(1,cs) ) ** 2  + ( self.Snx(1,cs) ) ** 2 ).unsqueeze(0)
        for k in range(2,n_modes):
            Z_m = torch.cat((Z_m, ( ( self.Cnx(k,cs) ) ** 2  + ( self.Snx(k,cs) ) ** 2 ).unsqueeze(0)), dim=0)  
        return torch.max( 2 * torch.cumsum(Z_m,0) / len(cs) - 4 * torch.tensor([m for m in range(1,n_modes)]).cuda() + 4 ) / len(cs)

    def Snx(self, n, cs):
        Sn = 0
        for k in range(1,n+2,2):
            Sn += ( (-1)**((k-1)/2) * scipy.special.binom(n,k) * cs[:,0]**(n-k) * cs[:,1]**k ).sum(0)
        return Sn
    def Cnx(self, n, cs):
        Cn = 0
        for k in range(0,n+2,2):
            Cn += ( (-1)**(k/2) * scipy.special.binom(n,k) * cs[:,0]**(n-k) * cs[:,1]**k ).sum(0)
        return Cn


class I0Function(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input):
        #preprocess
        input_cpu = input.data.cpu().numpy()
        #full loss
        output = i0(input_cpu)
        grad = i1(input_cpu)
        ctx.save_for_backward(torch.Tensor(grad).to(input.device)) #caching these for backwards pass
        return torch.Tensor(output).to(input.device)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        grad = ctx.saved_tensors[-1]
        # input_cpu = input.data.cpu().numpy()
        # if ctx.needs_input_grad[0]:
        #     grad_input = i1(input_cpu)

        return grad_output * grad #torch.Tensor(grad_input).to(input.device)

class I0(nn.Module):
    def __init__(self,):
        super(I0, self).__init__()

    def forward(self, input):
        return I0Function.apply(input)

besseli0 = I0()

class MSErrLossAll2(nn.Module):
    """Angle between input and target vectors, where input vector is an x,y point on the unit circle corresponding
    to input angle, and target vector has x,y points in [-1,1] x [-1,1] (L_Inf ball).

    Only the angle is used for optimization (for now), but the magnitude of the target vector also conveys some
    information - the certainty of the prediction/eccentricity of the system. Can report this (or a normalized
    version that accounts for the max length that's different from the axes vs the corners due to anisotropy).
    """
    # 0.4*self.lambda_E * F.smooth_l1_loss(2.5*input[:,5],2.5*energy,reduction='none') 
    def __init__(self, size_average=True, reduce=True, alpha=1, Z=None, lambda_abs=1, lambda_E=1):
        super(MSErrLossAll2, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.alpha = alpha
        self.Z = Z
        self.lambda_abs = lambda_abs
        self.lambda_E = lambda_E

    def forward(self, input, target):
        # Works when target is broadcastable
        norm = input[:,:2] / torch.sqrt(input[:,0]**2 + input[:,1]**2).unsqueeze(1)
        abs_pt = target[:,2:4]
        energy = target[:,4]
        loss = 0.5 * ((norm - target[:,:2])**2).sum(1) * self.alpha \
              - torch.exp(-input[:,2]) * ( (norm[:,0]**2 - norm[:,1]**2) * (target[:,0]**2 - target[:,1]**2) + 2*norm[:,0]*norm[:,1] * 2*target[:,0]*target[:,1] ) \
              + self.lambda_abs * ((input[:,3:5] - abs_pt)**2).sum(1) + 0.4 * self.lambda_E *  ((input[:,5] - energy)**2) \
              + torch.log(besseli0(torch.exp(-input[:,2])))

        if self.reduce:
            if self.size_average:
                loss = torch.mean(loss)
            else:
                loss = torch.sum(loss)

        return loss

class MSErrLossAllE(nn.Module):
    """Angle between input and target vectors, where input vector is an x,y point on the unit circle corresponding
    to input angle, and target vector has x,y points in [-1,1] x [-1,1] (L_Inf ball).

    Only the angle is used for optimization (for now), but the magnitude of the target vector also conveys some
    information - the certainty of the prediction/eccentricity of the system. Can report this (or a normalized
    version that accounts for the max length that's different from the axes vs the corners due to anisotropy).
    """
    def __init__(self, size_average=True, reduce=True, alpha=1):
        super(MSErrLossAllE, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.alpha = alpha

    def forward(self, input, target):
        # Works when target is broadcastable
        norm = input[:,:2] / torch.sqrt(input[:,0]**2 + input[:,1]**2).unsqueeze(1)
        abs_pt = target[:,2:4]
        energy = target[:,4]
        alpha = (energy * 4.5 + 1 - 1.5) / 2
        loss = 0.5 * torch.exp(-input[:,2]) * ( alpha * ((norm - target[:,:2])**2).sum(1) #) + 0.5 * input[:,2]  
              + ( ( (norm[:,0]**2 - norm[:,1]**2) - (target[:,0]**2 - target[:,1]**2) )**2 + (2*norm[:,0]*norm[:,1] - 2*target[:,0]*target[:,1])**2 ) ) / (1 + alpha) \
              + ((input[:,3:5] - abs_pt)**2).sum(1) + ((input[:,5] - energy)**2) \
              + 0.5 * input[:,2]

        if self.reduce:
            if self.size_average:
                loss = torch.mean(loss)
            else:
                loss = torch.sum(loss)
        return loss

def main():
    return 0

if __name__ == '__main__':
    main()
