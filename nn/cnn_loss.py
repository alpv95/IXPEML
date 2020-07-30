# Convolutional neural net image processing for inner/individual track angle prediction
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import scipy


class AngularLoss(nn.Module):
    """L1 loss on angle distance; shorter leg. Targets are provided as angles in radians in [-pi,pi). Inputs
    are coerced to the same range using modular arithmetic.
    """
    def __init__(self, size_average=True, reduce=True):
        super(AngularLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, input, target):
        # Rotate so input is 0 rad. Then the code below is easy to understand.
        x = input - target
        y = target - target

        # Wrap into [-pi,pi)
        # Source: https://stackoverflow.com/questions/11498169/dealing-with-angle-wrap-in-c-code
        x = torch.fmod(x + np.pi, 2*np.pi)
        x = x + (x < 0).float() * 2*np.pi
        x = x - np.pi

        # Angular distance is then the simple difference
        # Note that it's always positive. The gradient reveals the right direction? Yes, but the magnitude
        #   of the gradient is always 1?
        loss = F.l1_loss(x, y, reduce=False)

        if self.reduce:
            if self.size_average:
                loss = torch.mean(loss)
            else:
                loss = torch.sum(loss)
        return loss


class CosineLoss(nn.Module):
    """Angle between input and target vectors, where input vector is an x,y point on the unit circle corresponding
    to input angle, and target vector has x,y points in [-1,1] x [-1,1] (L_Inf ball).

    Only the angle is used for optimization (for now), but the magnitude of the target vector also conveys some
    information - the certainty of the prediction/eccentricity of the system. Can report this (or a normalized
    version that accounts for the max length that's different from the axes vs the corners due to anisotropy).
    """
    def __init__(self, size_average=True, reduce=True):
        super(CosineLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, input, target):
        # Works when target is broadcastable
        loss = 1 - F.cosine_similarity(input, target)

        if self.reduce:
            if self.size_average:
                loss = torch.mean(loss)
            else:
                loss = torch.sum(loss)
        return loss

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

class MAErrLoss(nn.Module):
    """Angle between input and target vectors, where input vector is an x,y point on the unit circle corresponding
    to input angle, and target vector has x,y points in [-1,1] x [-1,1] (L_Inf ball).

    Only the angle is used for optimization (for now), but the magnitude of the target vector also conveys some
    information - the certainty of the prediction/eccentricity of the system. Can report this (or a normalized
    version that accounts for the max length that's different from the axes vs the corners due to anisotropy).
    """
    def __init__(self, size_average=True, reduce=True, alpha=1):
        super(MAErrLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.alpha = alpha

    def forward(self, input, target):
        # Works when target is broadcastable
        norm = input[:,:2] / torch.sqrt(input[:,0]**2 + input[:,1]**2).unsqueeze(1)
        loss = 0.5 * torch.exp(-input[:,2]) * ( (abs(norm - target) ).sum(1) #) + 0.5 * input[:,2]  
              + self.alpha * ( abs( (norm[:,0]**2 - norm[:,1]**2) - (target[:,0]**2 - target[:,1]**2) ) + abs(2*norm[:,0]*norm[:,1] - 2*target[:,0]*target[:,1]) ) ) \
              + 0.5 * input[:,2]

        if self.reduce:
            if self.size_average:
                loss = torch.mean(loss)
            else:
                loss = torch.sum(loss)
        return loss

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
        norm1 = input[:,:2] / torch.sqrt(input[:,0]**2 + input[:,1]**2).unsqueeze(1)
        norm2 = input[:,3:5] / torch.sqrt(input[:,3]**2 + input[:,4]**2).unsqueeze(1)        
        abs_pt = target[:,4:6]
        energy = target[:,6]
        loss = ( 0.5 * torch.exp(-input[:,2]) * ( ((norm1 - target[:,:2])**2).sum(1) ) + 0.5 * input[:,2]  
               + (0.5 * torch.exp(-input[:,5]) * ( ((norm2 - target[:,2:4])**2).sum(1) ) + 0.5 * input[:,5]) * self.alpha ) \
              + self.lambda_abs * ((input[:,6:8] - abs_pt)**2).sum(1) + self.lambda_E * ((input[:,8] - energy)**2)

        if self.reduce:
            if self.size_average:
                loss = torch.mean(loss)
            else:
                loss = torch.sum(loss)
        if self.Z is not None:
            return loss + self.Z * self.Z_m(norm1)
        else:
            return loss

    def Z_m(self, cs):
        n_modes = 3
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
    sep = '===================================================='

    # Test out loss functions for predicting an angle
    #   With the periodicity
    # Create 2 angles and their corresponding points
    #   dim 1 is dummy batch
    #   dim 2 is vals
    p1a = np.array([[0]], dtype=np.float32)  # Vary this to check
    p1as = np.tile(p1a, [8, 1])  # for matching p2as
    p1 = np.array([[1, 0]], dtype=np.float32)  # along pos x-axis

    p2a = np.array([[np.pi/4]], dtype=np.float32)
    p2 = np.array([[0.5, 0.5]], dtype=np.float32)

    # p2 = np.array([[0, 1]], dtype=np.float32)  # along pos y-axis, their angular distance is pi

    p2as = np.array([[np.pi/4],  # range is [0,2pi)
                     [np.pi/2],
                     [3/4*np.pi],
                     [np.pi],
                     [5/4*np.pi],
                     [3/2*np.pi],
                     [7/4*np.pi],
                     [0]], dtype=np.float32)
    # p2as = np.array([[np.pi/4],  # range is [-pi,pi)
    #                  [np.pi/2],
    #                  [3/4*np.pi],
    #                  [-np.pi],
    #                  [-3/4*np.pi],
    #                  [-np.pi/2],
    #                  [-np.pi/4],
    #                  [0]], dtype=np.float32)
    p2s = np.array([[0.5, 0.5],  # pts around the circle
                    [0, 1],
                    [-0.5, 0.5],
                    [-1, 0],  # farthest away
                    [-0.5, -0.5],
                    [0, -1],
                    [0.5, -0.5],
                    [1, 0]], dtype=np.float32)

    # input = p2as
    # target = 0
    # d = angular_distance(input, target)

    p1at = torch.from_numpy(p1a)
    p1ast = torch.from_numpy(p1as)
    p1t = torch.from_numpy(p1)
    p2at = torch.from_numpy(p2a)
    p2t = torch.from_numpy(p2)
    p2ast = torch.from_numpy(p2as)
    p2st = torch.from_numpy(p2s)

    p1av = Variable(p1at, requires_grad=False)
    p1asv = Variable(p1ast, requires_grad=False)
    p1v = Variable(p1t, requires_grad=False)
    p2av = Variable(p2at, requires_grad=True)
    p2av.register_hook(print)
    p2v = Variable(p2t, requires_grad=True)
    p2v.register_hook(print)
    p2asv = Variable(p2ast, requires_grad=True)
    p2sv = Variable(p2st, requires_grad=True)

    print(p1t)
    print(p2t)
    print(p2st)

    print(sep)

    # 1) Linear distance of angles
    #   Naive method that doesn't account for distance on circle
    print('LIN' + sep)
    loss1 = nn.L1Loss(reduce=False)
    out1 = loss1(p2asv, p1asv)
    print(out1)

    loss1_ = nn.L1Loss()
    out1_ = loss1_(p2av, p1av)
    out1_.backward()

    print(sep)

    # 2) Closest angular distance
    print('ANG' + sep)
    loss2 = AngularLoss(reduce=False)
    out2 = loss2(p2asv, p1asv)
    print(out2)

    loss2_ = AngularLoss()
    out2_ = loss2_(p2av, p1av)
    out2_.backward()

    print(sep)

    # 3) Cosine distance
    print('COS' + sep)
    # cos = nn.CosineSimilarity()  # keep default dim=1, eps=1e-8
    # d3t = cos(p1t, p2t)  # should be 0 for orthogonal
    # d3ts = 1 - F.cosine_similarity(p1t, p2ts)
    # print(d3ts)

    loss3 = CosineLoss(reduce=False)
    out3 = loss3(p2sv, p1v)
    print(out3)

    loss3_ = CosineLoss()
    out3_ = loss3_(p2v, p1v)
    out3_.backward()
    # print(out3_)

    return 0


if __name__ == '__main__':
    main()
