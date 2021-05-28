'''
Class with framework to run individual NNs and NN ensembles on simulated and measured track datasets.
Inlcudes weighting and cut polarization analyses.
'''

import torch
import numpy as np
import pickle
import h5py
import sys, os
from util.pydataloader import H5Dataset, H5DatasetEval, ZNormalize
from collections import namedtuple
from nn.cnn import TrackAngleRegressor
from torchvision import transforms
from astropy.io import fits
from util.methods import *

class NetTest(object):
    """Interface for testing trained networks on measured or simulated data"""
    base = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    model_base = os.path.join(base, "data/nn", "")
    data_base =  os.path.join(base, "data","")
    save_base =  base 

    def __init__(self, nets=[], datasets=[], n_nets=1, datatype="sim",
                save_table=None, input_channels=2, stokes_correct=None, batch_size=2048, 
                nworkers=1):
 
        self.nets = [os.path.join(self.model_base,net) for net in nets]
        self.datasets = [os.path.join(self.data_base,data) for data in datasets]
        self.datatype = datatype
        self.n_nets = n_nets
        self.batch = batch_size
        self.nworkers = nworkers
        self.save_table = save_table
        self.input_channels = input_channels
        self.stokes_correct = stokes_correct


    def _predict(self, net, dataset, augment=3): #base single net on single dataset prediction
        '''
        Predicts NN track angles with NN 'net' on 'dataset'.
        Requires GPU for fast computation.
        '''
        datatype = self.datatype
        up2 = lambda p: os.path.dirname(os.path.dirname(p))

        if "peakonly" in net:
            self.losstype = 'mserrall3'
            batch_factor = 1
        elif "tailvpeak" in net:
            self.losstype = 'tailvpeak'
            batch_factor = 2
        
        mean, std = torch.load(os.path.join(up2(net),"ZN.pt"))
        meanE, stdE = torch.load(os.path.join(up2(net),"ZNE.pt"))

        dataset = H5DatasetEval(dataset, datatype=datatype, losstype=self.losstype, energy_cal=(meanE, stdE),
                                    augment=augment, transform=transforms.Compose([ZNormalize(mean=mean,std=std)]))

        kwargs = {'num_workers': self.nworkers, 'pin_memory': True}
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch*batch_factor, shuffle=False, **kwargs)
           
        m = TrackAngleRegressor(self.losstype, load_checkpoint=net,)

        y_hats = m.predict(test_loader)
        angles_mom = (dataset.mom_phis).numpy()
        mom_abs_pts = np.reshape( (dataset.mom_abs_pts).numpy(), (-1,2), order="C")
        p_tail = [None]
        if self.losstype == 'mserrall3':
            errors = y_hats[1,:]
            abs_pts = np.transpose(y_hats[2:4,:])
            energies = y_hats[4,:]
            angles = y_hats[0,:]
        else:
            p_tail = y_hats
            angles = [None]
            errors = [None]
            abs_pts = [None]
            energies = [None]
                
        try:
            moms = dataset.moms.numpy()
        except AttributeError:
            moms = dataset.moms
        
        if datatype =='sim':
            angles_mom = np.ndarray.flatten( angles_mom, "C" )
            moms = np.ndarray.flatten( moms, "C" )
            zs = np.ndarray.flatten( dataset.zs.numpy(), "C" )
            angles_sim = np.ndarray.flatten( dataset.angles.numpy(), "C" )
            abs_pts_sim = (dataset.abs_pts).numpy()
            abs_pts_sim = np.reshape( abs_pts_sim, (-1,2), order="C" )
            energies_sim = (dataset.energy).numpy()
            energies_sim = np.ndarray.flatten( energies_sim, "C" )
            trgs = [None]
            flags = dataset.flags.numpy() 
        else:
            trgs = dataset.trgs.numpy()
            flags = dataset.flags.numpy()
            angles_sim = [None]
            abs_pts_sim = [None]
            energies_sim = [None]
            zs = [None]
        xy_abs_pts = dataset.xy_abs_pts.numpy()

        #transform predicted energies back to kev if not none
        if energies[0] is not None:
            energies = energies * stdE.item() + meanE.item() 
        if energies_sim[0] is not None:
            energies_sim = np.round(energies_sim * stdE.item() + meanE.item(), 3)
        energies_mom = (dataset.mom_energy).numpy()

        return angles, angles_mom, angles_sim, moms, errors, abs_pts, mom_abs_pts, abs_pts_sim, energies, energies_sim, energies_mom, zs, trgs, flags, p_tail, xy_abs_pts

    def stokes_correction(self, angles):
        '''
        Corrects predicted photoelectron angles for spurious modulation by removing measured Stokes' parameters from calibration maps. 
        '''
        anchors_hex = angles[-1] #xy_abs_pts
        anchors_sq = angles[6] #mom_abs_pts
        abs_pts = angles[5] #abs_pts NN

        angles = list(angles)

        with fits.open(self.base + "real_data/systematic_UnpFF_DU2.fits", memmap=False) as hdu:
            Q = hdu[1].data
            U = hdu[2].data
            ENE = hdu[3].data
        energy_idx = np.argmin(abs(ENE - self.stokes_correct)) #which energy map to use
        print("Using Stokes energy map at {}keV\n".format(ENE[energy_idx]))

        abs_pts_sq = anchors_hex
        abs_pts_sq = np.reshape(abs_pts_sq, (-1, 2), order="C")
        print(abs_pts_sq.shape)
        for i,angle in enumerate(angles[:2]):
            #TODO: add in NN abs_pts

            bins = np.linspace(-7,7,100)
            x_grid = np.digitize(abs_pts_sq[:,0], bins)
            y_grid = np.digitize(abs_pts_sq[:,1], bins)
            
            Q_init = np.cos(2*angle)
            U_init = np.sin(2*angle)
            Q_final = Q_init + (Q[energy_idx,x_grid,y_grid] + Q[energy_idx,x_grid+1,y_grid] + Q[energy_idx,x_grid,y_grid+1] + Q[energy_idx,x_grid+1,y_grid+1] + Q[energy_idx,x_grid+1,y_grid-1] 
                                + Q[energy_idx,x_grid-1,y_grid+1] + Q[energy_idx,x_grid-1,y_grid] + Q[energy_idx,x_grid,y_grid-1] + Q[energy_idx,x_grid-1,y_grid-1]) / 9
            U_final = U_init + (U[energy_idx,x_grid,y_grid] + U[energy_idx,x_grid+1,y_grid] + U[energy_idx,x_grid,y_grid+1] + U[energy_idx,x_grid+1,y_grid+1] + U[energy_idx,x_grid+1,y_grid-1] 
                                + U[energy_idx,x_grid-1,y_grid+1] + U[energy_idx,x_grid-1,y_grid] + U[energy_idx,x_grid,y_grid-1] + U[energy_idx,x_grid-1,y_grid-1]) / 9

            angle_corrected = 0.5 * np.arctan2(U_final, Q_final)
            angles[i] = angle_corrected

        return tuple(angles)


    def ensemble_predict(self,):
        '''
        Goes through list of NNs in chosen ensemble and applies them to chosen datasets. Results are combined and saved in pickle file.
        Final mus and EVPAs for whole datasets are printed at the end.
        '''
        for data in self.datasets:
            name = data.replace(self.data_base,"") + "__" + "ensemble"
            results = ([],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[])
            for i, net in enumerate(self.nets[:1]):
                print(">> Angular NN {}/{} : \n".format(i+1,len(self.nets[:1])))
                results = tuple(map( np.append, results, self._predict(net, data, augment=6) ))
                print(">> Complete")
            if self.stokes_correct:
                results = self.stokes_correction(results)
            #Post processing for rotations and reducing repeated moments outputs
            results = post_rotate(results, len(self.nets[:1]), aug=6, datatype=self.datatype, losstype=self.losstype)

            results_ptail = ([],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[])
            for i, net in enumerate(self.nets[1:]):
                print(">> Ptail NN {}/{} : \n".format(i+1,len(self.nets[1:])))
                results_ptail = tuple(map( np.append, results_ptail, self._predict(net, data, augment=3) ))
                print(">> Complete")

            p_tail = triple_angle_reshape(results_ptail[-2], len(self.nets[1:]), augment=3)
            p_tail = np.mean(p_tail, axis=(1,2))
 
            if self.save_table is not None:
                name = data.replace(self.data_base,"") + "__" + self.save_table + "__" + "ensemble"
                fits_save(results, p_tail, os.path.join(self.save_base, name.replace("/","_")), self.datatype, self.losstype)
                print("Saved to: {}".format(os.path.join(self.save_base, name.replace("/","_") + ".fits"))) 






