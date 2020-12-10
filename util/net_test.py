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
from util.pydataloader import H5Dataset, ToTensor, ZNormalize, SelfNormalize
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
    data_base =  os.path.join(base, "data/expanded","")
    save_base =  base 
    plot_base = base

    def __init__(self, nets=[], datasets=[], fitmethod="stokes", plot=False, n_nets=1, cut=0.815, datatype="sim",
                save_table=None, input_channels=1, stokes_correct=None):
 
        self.method = fitmethod
        self.plot = plot
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
             losstype = f['root']['hparams']['losstype'][()].decode("utf-8")

        mean, std = torch.load(os.path.join(up2(net),"ZN.pt"))
        meanE, stdE = torch.load(os.path.join(up2(net),"ZNE.pt"))

        dataset = H5Dataset(dataset, datatype=datatype, losstype=losstype, energy_cal=(meanE, stdE),
                                    transform=transforms.Compose([ZNormalize(mean=mean,std=std)]))

        kwargs = {'num_workers': 4, 'pin_memory': True}
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs)
           
        m = TrackAngleRegressor(load_checkpoint=net, input_channels=self.input_channels)

        angles = m.predict(test_loader,bayes=bayes)
        angles_mom = (dataset.mom_phis).numpy()
        if angles.shape[0] == 2:
            errors = angles[1,:]
            angles = angles[0,:]
            mom_abs_pts = [None]
            abs_pts = [None]
            energies = [None]
        elif angles.shape[0] == 5:
            errors = angles[1,:]
            mom_abs_pts = np.reshape( (dataset.mom_abs_pts).numpy(), (-1,2), order="C")
            abs_pts = np.transpose(angles[2:4,:])
            energies = angles[4,:]
            angles = angles[0,:]
        else:
            errors = [None]
            abs_pts = [None]
            energies = [None]
            mom_abs_pts = [None]
                
        try:
            moms = dataset.moms.numpy()
        except AttributeError:
            moms = dataset.moms
        
        if datatype =='sim':
            augment = angles_mom.shape[1]
            angles_mom = np.ndarray.flatten( angles_mom, "C" )
            moms = np.ndarray.flatten( moms, "C" )
            zs = np.ndarray.flatten( dataset.zs.numpy(), "C" )
            angles_sim = np.ndarray.flatten( torch.atan2(dataset.angles[:,:,1],dataset.angles[:,:,0]).numpy(), "C" )
            abs_pts_sim = (dataset.abs_pts).numpy()
            abs_pts_sim = np.reshape( abs_pts_sim, (-1,2), order="C" )
            energies_sim = (dataset.energy).numpy()
            energies_sim = np.ndarray.flatten( energies_sim, "C" ) 
        else:
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

        return angles, angles_mom, angles_sim, moms, errors, abs_pts, mom_abs_pts, abs_pts_sim, energies, energies_sim, zs, xy_abs_pts

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

    def _mu_error_stokes(self,Q,U,mu,phi,N,net=False):
        net_factor = 1
        if net:
            net_factor = self.n_nets * 3
        return np.sqrt( 1/((N /net_factor)*(Q**2 + U**2)) * ( (0.5 - mu**2/4 * np.cos(2*phi)**2)*4*Q**2 + 
            (0.5 - mu**2/4 * np.sin(2*phi)**2)*4*U**2 - (mu**2/8 * np.sin(4*phi))*8*Q*U ) ) 

    def _phi_error_stokes(self,Q,U,mu,phi,N,net=False):
        net_factor = 1
        if net:
            net_factor = self.n_nets * 3
        return np.sqrt( 1/(4*(N / net_factor)*(Q**2 + U**2)**2) * ( (0.5 - mu**2/4 * np.cos(2*phi)**2) + 
            (0.5 - mu**2/4 * np.sin(2*phi)**2)*Q**4 + (mu**2/8 * np.sin(4*phi))*Q**2 ) )

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

    def _error_cut(self,angles,errors):
        '''
        Applies error based cuts to predicted photoelectron angles.
        '''
        mom_cuts = np.linspace(0.5,2.8,60)[::-1]
        for mom_cut in mom_cuts:
            if len(angles[errors < mom_cut]) < self.ellipticity_cut * len(angles):
                print("Moment cut at {:.2f}, leaving {:.3f} of dataset\n".format(mom_cut, len(angles[errors < mom_cut]) / len(angles)))
                break
        return mom_cut

    def Z_m(self, cs):
    	Z_m = ( ( self.Cnx(1,cs) ) ** 2  + ( self.Snx(1,cs) ) ** 2 ).unsqueeze(0)
    	for k in range(2,11):
            Z_m = torch.cat((Z_m, ( ( self.Cnx(k,cs) ) ** 2  + ( self.Snx(k,cs) ) ** 2 ).unsqueeze(0)), dim=0)  
    	return torch.max( 2 * torch.cumsum(Z_m,0) / len(cs) - 4 * torch.tensor([m for m in range(1,11)]) + 4 )

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

    def fit_mod(self, angles, method="stokes", error_weight=1):
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
        _, _, _, moms,errors = angles[:5]
        print("Method: {}\n".format(method))

        for i,angle in enumerate(angles[:3]):
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


    def plot_save(self, angles, mu, phi0, name=None, n_bins=100):
        bins = np.linspace(-np.pi, np.pi, n_bins)
        x = np.linspace(-np.pi, np.pi, 199)

        plt.figure()
        plt.xlabel('$\phi$ (rad)')
        plt.ylabel('Prob')
        #plt.title('Quality cut: {:.3f}; {}'.format(cut, name))
        plt.title(name)
        names = ['ML', 'Mom', 'Sim']
        for i, m in enumerate(mu):
            if angles[i] is not None:
                plt.hist(angles[i], bins=bins, density=True, alpha=0.3)
                N = (1 + m * np.cos(2*(x - phi0[i]))) / (2*np.pi)
                plt.plot(x, N, linewidth=2.3, label=r'{} $\phi_0$ = {:.3f}; $\mu$ = {:.3f}'.format(names[i], phi0[i], m))

        plt.legend()
        print("Saved plot: {}\n".format(name.replace("/","_") + '.pdf'))
        plt.savefig(self.plot_base + name.replace("/","_") + '.pdf', bbox_inches='tight') 

    
    def plot_tracks(self, ):
        for net in self.nets:
            for data in self.datasets:
                name = data.replace(self.data_base,"") + "__" + net.replace(self.model_base,"")
                angles = self._predict(net, data)
                self.results[name] = angles


    def single_predict(self, bayes=False):
        for net in self.nets:
            self.results[net.replace(self.model_base,"")] = []
            for data in self.datasets:
                name = data.replace(self.data_base,"") + "__" + net.replace(self.model_base,"")
                print("Evaluating Dataset__Net: {} \n".format(name))
                angles = self._predict(net, data, bayes)
                #SAVE predictions to test
                #pickle.dump(angles, open(self.base + name.replace("/","_") + ".pickle","wb"))
                print("Bayes Dropout Sampling: {}\n".format(bayes))
                
                if self.plot:
                   self.plot_save( angles, mu, phi0, name=name )
                #self.results[name] = (mu, phi0)
                if angles[2] is not None:
                    angle_err_weighted2 = np.mean( ((np.cos(2*angles[0]) - np.cos(2*angles[2]))**2 + (np.sin(2*angles[0]) - np.sin(2*angles[2]))**2) 
                                            / angles[4] ) * np.sum(angles[4]) / len(angles[4])
                    angle_err_weighted1 = np.mean( ((np.cos(angles[0]) - np.cos(angles[2]))**2 + (np.sin(angles[0]) - np.sin(angles[2]))**2) 
                                            / angles[4] ) * np.sum(angles[4]) / len(angles[4])
                    angle_err2 = np.mean((np.cos(2*angles[0]) - np.cos(2*angles[2]))**2 + (np.sin(2*angles[0]) - np.sin(2*angles[2]))**2)
                    angle_err1 = np.mean((np.cos(angles[0]) - np.cos(angles[2]))**2 + (np.sin(angles[0]) - np.sin(angles[2]))**2)
                    mom_angle_err2 = np.mean((np.cos(2*angles[1]) - np.cos(2*angles[2]))**2 + (np.sin(2*angles[1]) - np.sin(2*angles[2]))**2)
                    mom_angle_err1 = np.mean((np.cos(angles[1]) - np.cos(angles[2]))**2 + (np.sin(angles[1]) - np.sin(angles[2]))**2)
                    abs_pts_err = np.mean((angles[5][0] - angles[7][0])**2 + (angles[5][1] - angles[7][1])**2)
                    mom_abs_pts_err = np.mean((angles[6][0] - angles[7][0])**2 + (angles[6][1] - angles[7][1])**2)
                    energy_err = np.mean((angles[8] - angles[9])**2)
                    print("Accuracies:\n weighted_angle2: {:.3f} | angle2: {:.3f} | mom_angle2: {:.3f} | weighted_angle1: {:.3f} | angle1: {:.3f} | mom_angle1: {:.3f} | abs_pts: {} | mom_abs_pts: {} | energy: {}\n".format(
                                angle_err_weighted2, angle_err2, mom_angle_err2, angle_err_weighted1, angle_err1, mom_angle_err1, abs_pts_err, mom_abs_pts_err, energy_err))
                print("{}:\n mu + err | phi0 + err\n".format(name))
                angles = post_rotate(angles, 1, aug=3, fix=False, datatype=self.datatype)
                mu, phi0, mu_err, phi0_err = self.fit_mod(angles, method='stokes')
                mu_w, phi0_w, mu_err_w, phi0_err_w = self.fit_mod(angles, method="weighted_MLE")

                for i,_ in enumerate(mu):
                    print("{:.3f} +- {:.3f} | {:.3f} +- {:.3f}\n".format(mu[i],mu_err[i],phi0[i],phi0_err[i]))
                    print("{:.3f} +- {:.3f} | {:.3f} +- {:.3f} (weighted)\n".format(mu_w[i],mu_err_w[i],phi0_w[i],phi0_err_w[i]))
                cs = torch.stack([torch.cos(torch.from_numpy(angles[0])),torch.sin(torch.from_numpy(angles[0]))], axis=1)
                self.results[net.replace(self.model_base,"")].extend((self.Z_m(cs).item(),angle_err2,angle_err_weighted2, angle_err1, angle_err_weighted1, mu[0], mu_w[0], abs_pts_err, mom_abs_pts_err, energy_err))

        if self.save_table is not None:
            df = pd.DataFrame.from_dict(self.results,orient='index').transpose()
            df.to_pickle(os.path.join(self.save_base, self.save_table + ".pickle"))
            print("Saved to: {}".format(os.path.join(self.save_base, self.save_table + ".pickle")))
            print("SAVED!")


    def ensemble_predict(self, bayes=False):
        '''
        Goes through list of NNs in chosen ensemble and applies them to chosen datasets. Results are combined and saved in pickle file.
        Final mus and EVPAs for whole datasets are printed at the end.
        '''
        for data in self.datasets:
            name = data.replace(self.data_base,"") + "__" + "ensemble"
            results = ([],[],[],[],[],[],[],[],[],[],[],[])
            for i, net in enumerate(self.nets):
                print(">> NN {}/{} : \n".format(i+1,len(self.nets)))
                results = tuple(map( np.append, results, self._predict(net, data, bayes) ))
                print(">> Complete")
            if self.stokes_correct:
                results = self.stokes_correction(results)

            #Post processing for rotations and reducing repeated moments outputs
            results = post_rotate(results, self.n_nets, aug=3, datatype=self.datatype)

            mu, phi0, mu_err, phi0_err = self.fit_mod(results, method=self.method)
            mu_w, phi0_w, mu_err_w, phi0_err_w = self.fit_mod(results, method="weighted_MLE")
 
            if self.save_table is not None:
                name = data.replace(self.data_base,"") + "__" + self.save_table + "__" + "ensemble"
                fits_save(results, os.path.join(self.save_base, name.replace("/","_")), self.datatype)
                print("Saved to: {}".format(os.path.join(self.save_base, name.replace("/","_") + ".fits"))) 

            if self.plot:
                self.plot_save( results, mu, phi0, name )
            self.results[name] = (mu, mu_err, phi0_err, phi0)

            print(">> Modulation factors and EVPAs for entire dataset -->")
            print("{}:\n mu + err | phi0 + err\n".format(name))
            names = ["NN", "Mom.", "True"]
            algs = ["weighted", "cut", "n/a"]
            for i,_ in enumerate(mu):
                name = names[i]
                alg = algs[i]
                print(name + " {:.3f} +- {:.3f} | {:.3f} +- {:.3f}\n".format(mu[i],mu_err[i],phi0[i],phi0_err[i]))
                print(name + " {:.3f} +- {:.3f} | {:.3f} +- {:.3f} ({})\n".format(mu_w[i],mu_err_w[i],phi0_w[i],phi0_err_w[i], alg))




