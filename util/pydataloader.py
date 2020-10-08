'''
ALP torch.utils.data.Dataset class that allows for fast reading and shuffling of data for training and validation
Much more efficient than passing around huge numpy arrays
'''
import math
import h5py
import numpy as np
import os
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
#from formats.util import DenseSquareSimTracks

class H5Dataset(Dataset):
    def __init__(self, dirr, datatype='sim' ,losstype='ang', energy_cal=None, transform=None):
        #simple solution, just reading in as numpy array held in memory for now, H5 read too slow
        super(H5Dataset, self).__init__()
        #self.split = split
        self.datatype = datatype
        self.transform = transform
        self.dir = dirr
        n_pixels = 50

        print("Loading full dataset into memory\n")
        with open(self.dir + "tracks_full.pickle", "rb") as f:
            tracks_all = pickle.load(f)
        data_all = torch.load(self.dir + "labels_full.pt")
        print("Dataset size: ", data_all[0].shape)

        if datatype =='sim':
            self.tracks_cube = tracks_all
            self.angles = data_all[0]
            self.moms = data_all[1]
            self.xy_abs_pts = data_all[-1]

            #TODO: Save energy std and mean as fixed for the training dataset
            if energy_cal:
                #self.energy = (data_all[6] - torch.mean(data_all[6]) ) / torch.std(data_all[6]) #(data_all[6] - torch.min(data_all[6]) ) / (torch.max(data_all[6]) 
                            #- torch.min(data_all[6]) )
                self.energy = (data_all[5] - energy_cal[0]) / energy_cal[1]
            else:
                self.energy = (data_all[5] - torch.min(data_all[5]) ) / (torch.max(data_all[5]) - torch.min(data_all[5]) )

            self.mom_phis = data_all[2]
            self.abs_pts = torch.mean(data_all[3], axis=2) / n_pixels
            self.mom_abs_pts = torch.mean(data_all[4], axis=2) / n_pixels
            if (losstype == 'CE'):
                self.angles = self.angles.numpy()              
                self.angles = np.arctan2(self.angles[:,:,1], self.angles[:,:,0])
                self.angles = torch.from_numpy( np.digitize((180 / np.pi) * self.angles + 180, bins=np.linspace(60,360.01,6) ) )
            elif (losstype == "mserrall1" or losstype == "mserrall2"):
                self.angles = torch.stack((torch.cos(self.angles),torch.sin(self.angles),self.abs_pts[:,:,0], 
                                self.abs_pts[:,:,1], self.energy),2).float() #[batch_size, augment, 5] 
            # elif (losstype == "mserrall2"):
            #     self.angles = torch.stack((torch.cos(2*self.angles),torch.sin(2*self.angles), torch.cos(self.angles),torch.sin(self.angles),self.abs_pts[:,:,0], 
            #                     self.abs_pts[:,:,1], self.energy),2).float() #[batch_size, augment, 5] 
            elif (losstype == "mserr"):
                self.angles = torch.stack((torch.cos(self.angles),torch.sin(self.angles)),2).float()
        else:
            self.tracks_cube = tracks_all
            self.angles = torch.zeros(len(self.tracks_cube), 3, 5).float()
            self.moms = data_all[0]
            self.mom_phis = data_all[1]
            self.mom_abs_pts = torch.mean(data_all[2], axis=2) / n_pixels
            self.xy_abs_pts = data_all[-1]

        self.length = len(self.tracks_cube)
        self.pixels = n_pixels
    
    def __getitem__(self, index):
        sparse = self.tracks_cube[index]
        if sparse.shape[0] == 1:
            track = torch.stack([torch.sparse.FloatTensor(sparse[0,0,:2,:].long(), sparse[0,0,2,:], torch.Size([self.pixels,self.pixels])).to_dense(), 
                                torch.sparse.FloatTensor(sparse[0,1,:2,:].long(), sparse[0,1,2,:], torch.Size([self.pixels,self.pixels])).to_dense()])
        else:
            track = torch.stack([
                torch.stack([torch.sparse.FloatTensor(sparse[0,0,:2,:].long(), sparse[0,0,2,:], torch.Size([self.pixels,self.pixels])).to_dense(), 
                                torch.sparse.FloatTensor(sparse[0,1,:2,:].long(), sparse[0,1,2,:], torch.Size([self.pixels,self.pixels])).to_dense()]),
                torch.stack([torch.sparse.FloatTensor(sparse[1,0,:2,:].long(), sparse[1,0,2,:], torch.Size([self.pixels,self.pixels])).to_dense(), 
                                torch.sparse.FloatTensor(sparse[1,1,:2,:].long(), sparse[1,1,2,:], torch.Size([self.pixels,self.pixels])).to_dense()]),
                torch.stack([torch.sparse.FloatTensor(sparse[2,0,:2,:].long(), sparse[2,0,2,:], torch.Size([self.pixels,self.pixels])).to_dense(), 
                                torch.sparse.FloatTensor(sparse[2,1,:2,:].long(), sparse[2,1,2,:], torch.Size([self.pixels,self.pixels])).to_dense()])
            ])
        sample = (track.float(), self.angles[index])
        return sample
        
    def __len__(self):
        return self.length

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        track_image, angle = sample

        # expand dims to include channel dimension
        # torch image: C X H X W
        #track_image = np.expand_dims(track_image, axis=0)

        return (torch.from_numpy(track_image),
                torch.from_numpy(angle))

class ZNormalize(object):
    """Normalizes data to mean and std of training set"""
	
    def __init__(self, mean, std):
        self.mean = mean.float()
        self.std = std.float()
        #self.set_max = set_max

    def __call__(self, sample):
        track_image, angle = sample
        #track_image /= self.set_max # set all values to be floats between 0 and 1 
        track_image -= self.mean #centre data on mean per pixel
        return ( torch.where(self.std!=0, track_image / self.std, track_image), #make sure we dont divide by 0 if pixel std = 0
                  angle )

class SelfNormalize(object):
    """Normalizes data to mean and std of training set"""
	
    def __init__(self,):
        pass
        
    def __call__(self, sample):
        track_image, angle = sample
        maxx = torch.max(track_image)
        return ( track_image / maxx, #make sure we dont divide by 0 if pixel std = 0
                  angle )

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)
