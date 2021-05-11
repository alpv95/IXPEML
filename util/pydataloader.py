'''
torch.utils.data.Dataset class that allows for fast reading and shuffling of data for training and validation
Much more efficient than passing around huge numpy arrays
'''
import numpy as np
import os
import torch
import pickle
from torch.utils.data import Dataset, DataLoader

class H5Dataset(Dataset):
    def __init__(self, dirr, datatype='sim' ,losstype='ang', energy_cal=None, transform=None):
        super(H5Dataset, self).__init__()
        self.datatype = datatype
        self.transform = transform
        self.dir = dirr
        self.pixels = 50

        print("Loading full dataset into memory\n")
        with open(self.dir + "tracks_full.pickle", "rb") as f:
            self.tracks_cube = pickle.load(f)
        data_all = torch.load(self.dir + "labels_full.pt")

        self.moms = data_all["moms"]
        self.mom_energy = data_all["mom_energy"]
        self.mom_phis = data_all["mom_phis"]
        self.xy_abs_pts = data_all["xy_abs"]
        self.mom_abs_pts = torch.mean(data_all["mom_abs"], axis=2) / self.pixels
        self.length = len(self.tracks_cube)
        print("Dataset size: ", self.length)
        
        if datatype =='sim':
            assert "angles" in data_all.keys(), "--datatype should be set to 'meas', not 'sim'"
            self.angles = data_all["angles"]
            self.abs_pts = torch.mean(data_all["abs"], axis=2) / self.pixels
            self.zs = data_all["z"]

            if energy_cal:
                self.energy = (data_all["energy"] - energy_cal[0]) / energy_cal[1]
            else:
                raise("No energy calibration!")

            if (losstype == "mserrall1" or losstype == "mserrall2"):
                self.Y = torch.stack((torch.cos(self.angles),torch.sin(self.angles),self.abs_pts[:,:,0], 
                                self.abs_pts[:,:,1], self.energy),2).float() #[batch_size, augment, 5] 
            elif (losstype == "energy"):
                self.Y = self.energy.float()
            elif (losstype == "tailvpeak"):
                self.Y = torch.where(torch.lt(self.zs, 0.835) + torch.gt(self.zs, 10.83), torch.tensor(1), torch.tensor(0)).float()
            elif (losstype == "tailvpeak2"):
                self.Y = torch.where(torch.lt(self.zs, 0.835) + torch.gt(self.zs, 10.83), torch.tensor(1), torch.tensor(0))
                self.Y[torch.gt(self.zs, 10.83)] = torch.tensor(2, dtype=torch.long)
        else:
            self.trgs = data_all["trg_id"]
            self.flags = data_all["flag"]
            self.Y = None
    
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
        sample = (track.float(), self.Y[index])
        return sample
        
    def __len__(self):
        return self.length

class H5DatasetEval(H5Dataset):
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
        sample = track.float()
        return sample

class ZNormalize(object):
    """Normalizes data to mean and std of training set"""
	
    def __init__(self, mean, std):
        self.mean = mean.float()
        self.std = std.float()

    def __call__(self, sample):
        track_image, angle = sample
        track_image -= self.mean #centre data on mean per pixel
        return ( torch.where(self.std!=0, track_image / self.std, track_image), #make sure we dont divide by 0 if pixel std = 0
                  angle )
