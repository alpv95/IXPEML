# Convolutional neural net image processing for inner/individual track angle prediction
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import scipy

class MSErrLoss(nn.Module):
    """Angle between input and target vectors, where input vector is an x,y point on the unit circle corresponding
    to input angle, and target vector has x,y points in [-1,1] x [-1,1] (L_Inf ball).

    Only the angle is used for optimization (for now), but the magnitude of the target vector also conveys some
    information - the certainty of the prediction/eccentricity of the system. Can report this (or a normalized
    version that accounts for the max length that's different from the axes vs the corners due to anisotropy).
    """
    def __init__(self, size_average=True, reduce=True, alpha=1, Z = None):
        super(MSErrLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.alpha = alpha
        self.Z = Z

    def forward(self, input, target):
        # Works when target is broadcastable
        norm = input[:,:2] / torch.sqrt(input[:,0]**2 + input[:,1]**2).unsqueeze(1)
        loss = ( 0.5 * torch.exp(-input[:,2]) * ( ((norm - target[:,:2])**2).sum(1) ) * self.alpha
                + ( ( (norm[:,0]**2 - norm[:,1]**2) - (target[:,0]**2 - target[:,1]**2) )**2 + (2*norm[:,0]*norm[:,1] - 2*target[:,0]*target[:,1])**2 ) ) \
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
        n_modes = 3
        Z_m = ( ( self.Cnx(1,cs) ) ** 2  + ( self.Snx(1,cs) ) ** 2 ).unsqueeze(0)
        for k in range(2,n_modes):
            Z_m = torch.cat((Z_m, ( ( self.Cnx(k,cs) ) ** 2  + ( self.Snx(k,cs) ) ** 2 ).unsqueeze(0)), dim=0)  
        return torch.max( 2 * torch.cumsum(Z_m,0) / len(cs) - 4 * torch.tensor([m for m in range(1,n_modes)]).cuda() + 4 )

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
              + 0.5 * torch.exp(-input[:,2]) * (( ( (norm[:,0]**2 - norm[:,1]**2) - (target[:,0]**2 - target[:,1]**2) )**2 + (2*norm[:,0]*norm[:,1] - 2*target[:,0]*target[:,1])**2 ) ) \
              + self.lambda_abs * ((input[:,3:5] - abs_pt)**2).sum(1) + 0.4 * self.lambda_E * (F.smooth_l1_loss(2.5*input[:,5],2.5*energy,reduction='none') + 0.3*0.5*torch.abs(1 + torch.sign(input[:,5] - energy)) * (2.5*input[:,5] - 2.5*energy)**2) \
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
