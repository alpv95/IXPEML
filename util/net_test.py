'''
Class with framework to run individual NNs and NN ensembles on simulated and measured track datasets.
Inlcudes weighting and cut polarization analyses.
'''

import torch
import numpy as np
import pickle
import h5py
import scipy
from operator import add
import sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from util import loss
from util.pydataloader import H5Dataset, H5DatasetEval, ToTensor, ZNormalize, SelfNormalize
from collections import namedtuple
from nn.cnn import TrackAngleRegressor
from formats.dense_square import DenseSquareSimTracks
from formats.sparse_hex import SparseHexTracks, SparseHexSimTracks
from torchvision import transforms
from scipy.optimize import minimize 
import pandas as pd
from scipy.optimize import minimize_scalar
from astropy.io import fits
from util.methods import *

class NetTest(object):
    """Interface for testing trained networks on measured or simulated data"""
    base = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    model_base = os.path.join(base, "data/nn", "")
    data_base =  os.path.join(base, "data","")
    save_base =  base 
    plot_base = base

    def __init__(self, nets=[], datasets=[], fitmethod="stokes", n_nets=1, cut=0.815, datatype="sim",
                save_table=None, input_channels=2, stokes_correct=None):
 
        self.method = fitmethod
        self.nets = [os.path.join(self.model_base,net) for net in nets]
        self.datasets = [os.path.join(self.data_base,data) for data in datasets]
        self.results = {}
        self.datatype = datatype
        self.n_nets = n_nets
        self.ellipticity_cut = cut
        self.save_table = save_table
        self.input_channels = input_channels
        self.stokes_correct = stokes_correct


    def _predict(self, net, dataset, bayes=False,): #base single net on single dataset prediction
        '''
        Predicts NN track angles with NN 'net' on 'dataset'.
        Requires GPU for fast computation.
        '''
        datatype = self.datatype
        up2 = lambda p: os.path.dirname(os.path.dirname(p))

        with h5py.File(os.path.join(up2(net), 'opts.h5'),'r') as f:
             batch_size = 2048 #Can be adjusted if there are memory GPU memory problems during prediction
             self.losstype = f['root']['hparams']['losstype'][()].decode("utf-8")
        
        mean, std = torch.load(os.path.join(up2(net),"ZN.pt"))
        meanE, stdE = torch.load(os.path.join(up2(net),"ZNE.pt"))

        dataset = H5DatasetEval(dataset, datatype=datatype, losstype=self.losstype, energy_cal=(meanE, stdE),
                                    transform=transforms.Compose([ZNormalize(mean=mean,std=std)]))

        kwargs = {'num_workers': 4, 'pin_memory': True}
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs)
           
        m = TrackAngleRegressor(load_checkpoint=net, input_channels=self.input_channels)

        y_hats = m.predict(test_loader,bayes=bayes)
        angles_mom = (dataset.mom_phis).numpy()
        mom_abs_pts = np.reshape( (dataset.mom_abs_pts).numpy(), (-1,2), order="C")
        p_tail = [None]
        if y_hats.shape[0] >= 5:
            errors = y_hats[1,:]
            abs_pts = np.transpose(y_hats[2:4,:])
            energies = y_hats[4,:]
            angles = y_hats[0,:]
        else:
            print('TAILVPEAK')
            if self.losstype == 'energy':
                p_tail = y_hats * stdE.item() + meanE.item()
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
            flags = [None] 
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

    def _MLError(self, angles, mu_hat, phi_hat):
        denom = (1 + mu_hat*np.cos(2*(angles - phi_hat)))**2
        I00 = np.sum(np.cos(2*(angles - phi_hat))**2 / denom)
        I01 = np.sum(2*np.sin(2*(angles - phi_hat)) / denom)
        I11 = np.sum(4*mu_hat*(mu_hat + np.cos(2*(angles - phi_hat)) ) / denom)
        I = np.array([[I00,I01],[I01,I11]])
        I_1 = np.linalg.inv(I)
        return np.sqrt(I_1[0,0]), np.sqrt(I_1[1,1]), I_1[0,1]/np.sqrt(I_1[0,0]*I_1[1,1])
          
    def _mom_cut(self,angles,moms):
        '''
        Applies moment based cuts to predicted photoelectron angles.
        '''
        mom_cuts = np.arange(1,4,0.01)
        for mom_cut in mom_cuts:
            if len(angles[moms > mom_cut]) < self.ellipticity_cut * len(angles):
                print("Moment cut at {:.2f}, leaving {:.3f} of dataset\n".format(mom_cut, len(angles[moms > mom_cut]) / len(angles)))
                break
        return mom_cut


    def fit_mod(self, results, method="stokes", error_weight=1):
        '''
        Fits dataset of angles for modulation factor mu and EVPA phi0 with either standard Stokes method (Kislat et al.), weighted MLE (Peirson et al.) or ellipticity cuts.
        
        input:
        * angles -- tuple of lists containing photoelectron angles (from NN, moment analysis and truth if tracks simulated), moments and errors (angles_NN, angles_mom, angles_true, moments, errors)
        * method -- either 'stokes' or 'weighted_MLE', 'weighted_MLE' applies ellipticty cut stokes to the angles_mom
        * error_weight -- lambda parameters that controls weighting strength in 'weighted_MLE', see (Peirson et al.) 

        returns:
        * mu -- list of final modulation factors, [NN, moment analysis, truth]
        * phi0 -- list of final EVPAs, [NN, moment analysis, truth]
        * mu_err -- errors on mu, [NN, moment analysis, truth]
        * phi_err -- errors on phi0, [NN, moment analysis, truth]
        '''
        mu = []
        mu_err = []
        phi0 = []
        phi0_err = []
        _, _, _, moms,errors = results[:5]
        print("Method: {}\n".format(method))

        for i,angle in enumerate(results[:3]):
            if angle[0] is not None:
                if method == "stokes":
                    angle = np.ndarray.flatten(np.array(angle))
                    
                    N = len(angle)
                    Q = sum(np.cos(2*angle)) / N
                    U = sum(np.sin(2*angle)) / N
                
                    mu.append(2*np.sqrt(Q**2 + U**2))
                    phi0.append(np.arctan2(U,Q)/2) 

                    errmu, errphi0, _ = self._MLError(angle, mu[i], phi0[i])

                    mu_err.append(errmu)
                    phi0_err.append(errphi0)


                elif method == "weighted_MLE":

                    if errors[0] is None or i>0:
                        error = np.ones_like(angle)
                        moms = np.ndarray.flatten(np.array(moms))
                        mom_cut = self._mom_cut(angle, moms)
                        angle = angle[moms > mom_cut]
                        
                        N = len(angle)
                        Q = sum(np.cos(2*angle)) / N
                        U = sum(np.sin(2*angle)) / N   
                        mu.append(2*np.sqrt(Q**2 + U**2))
                        phi0.append(np.arctan2(U,Q)/2) 

                    else:
                        mu_holder, phi_holder, _ = weighted_stokes(angle, 1/errors, error_weight)
                        mu.append(mu_holder)
                        phi0.append(phi_holder)       

                    errmu, errphi0, _ = self._MLError(angle, mu[i], phi0[i])
                    mu_err.append(errmu)
                    phi0_err.append(errphi0)
                else:
                    raise("Method not recognized")

        return mu, phi0, mu_err, phi0_err 


    def ensemble_predict(self, bayes=False):
        '''
        Goes through list of NNs in chosen ensemble and applies them to chosen datasets. Results are combined and saved in pickle file.
        Final mus and EVPAs for whole datasets are printed at the end.
        '''
        for data in self.datasets:
            name = data.replace(self.data_base,"") + "__" + "ensemble"
            results = ([],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[])
            for i, net in enumerate(self.nets):
                print(">> NN {}/{} : \n".format(i+1,len(self.nets)))
                results = tuple(map( np.append, results, self._predict(net, data, bayes) ))
                print(">> Complete")
            if self.stokes_correct:
                results = self.stokes_correction(results)

            #Post processing for rotations and reducing repeated moments outputs
            results = post_rotate(results, self.n_nets, aug=3, datatype=self.datatype, losstype=self.losstype)
 
            if self.save_table is not None:
                name = data.replace(self.data_base,"") + "__" + self.save_table + "__" + "ensemble"
                fits_save(results, os.path.join(self.save_base, name.replace("/","_")), self.datatype, self.losstype)
                print("Saved to: {}".format(os.path.join(self.save_base, name.replace("/","_") + ".fits"))) 






