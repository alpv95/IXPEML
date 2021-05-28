'''
Takes fits files of GPDSW reconstructed hexagonal photoelectron tracks and assembles them into square tracks useable by NN ensembles.
For both real (measured) and simulated tracks. 
'''

import os
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from prep.convert_hex_to_square_tracks import hex2square
from util.split_data import tvt_random_split, tt_random_split
import torch
from scipy.stats import norm
from astropy.io import fits
import argparse
import pickle
from util.methods import *

parser = argparse.ArgumentParser()
parser.add_argument('input_file', type=str,
                    help='input folder or file')
parser.add_argument('out_base', type=str,
                    help='output folder')
parser.add_argument('--augment', type=int, choices=[1,3,6], default=3,
                    help='Number of Track augmentation randomly distributed in dataset. These should always be 3 during inference and 1 during training')
parser.add_argument('--npix', type=int, choices=[30, 50], default=50,
                    help='Number of pixels in square conversions. This should be 50 for >= v1.2')  
parser.add_argument('--sim', action='store_true',
                    help='Whether to incorporate monte carlo simulated data, not strictly necessary.')
parser.add_argument('--tot', type=int, default=None,
                    help='The total number of tracks to convert to square')
parser.add_argument('--peak_only', action='store_true',
                    help='No low or high Z tail tracks')
parser.add_argument('--tailvpeak', default=None, type=float,
                    help='Fraction of Peak tracks to keep.')
args = parser.parse_args()


# Specify options for datasets and their input -> output properties
# The output file is constructed from each row's name+type
Dataset = namedtuple('Dataset', ['file', 'total'])


class builder(object):
    in_file = args.input_file
    out_base = args.out_base
    n_pixels = args.npix
    augment = args.augment
    shift = 2
    peak_only = args.peak_only
    tailvpeak = args.tailvpeak

    def __init__(self, total):
        self.total = total
        self.out_files = []

        self.build_result = {"tracks": [], "mom_phis": torch.zeros( (self.total, self.augment) ), "moms": torch.zeros( self.total, self.augment ),
                             "mom_abs": torch.zeros( (self.total, self.augment, self.shift, 2) ), "xy_abs": torch.zeros( (self.total, 2) ), 
                             "mom_energy": torch.zeros( self.total), "flag": torch.zeros( self.total, dtype=torch.int16) }

    def init_build(self, input_file,):
        try:
            with fits.open(input_file.replace("_recon","_mc"), memmap=True) as hdu:
                data_mc = hdu['MONTE_CARLO'].data
            with fits.open(input_file, memmap=True) as hdu:
                data_events = hdu['EVENTS'].data
            assert 'DETPHI' in [c.name for c in data_events.columns], "Need *_recon.fits not *.fits as the measured input file"
            data = (data_events,data_mc)    
        except FileNotFoundError:
            print("No Monte Carlo, measured data only \n")
            with fits.open(input_file, memmap=True) as hdu:
                data = hdu['EVENTS'].data
            assert 'DETPHI' in [c.name for c in data.columns], "Need *_recon.fits not *.fits as the measured input file"

        return data

    def save(self, split, split_func):
        N_final = len(self.build_result["tracks"])
        if split[0] == 1:
            indxs = [np.arange(N_final)]
            self.out_files = [self.out_files[0]]
        else:
            indxs = split_func(np.arange(N_final), fracs=split)
        
        for indx, save_file in zip(indxs, self.out_files):
            tracks_cum_save = [torch.from_numpy(self.build_result["tracks"][idx]) for idx in indx]
            torch.save(tracks_cum_save, os.path.join(save_file, "tracks_full.pickle"))
            torch.save( {key: value[indx] for (key,value) in self.build_result.items() if key != "tracks"}, os.path.join(save_file,"labels_full.pt"))
            print("Saved, ", save_file )
        
        return indxs[0]

    def predict_energy(self, pha):
        """predicts track energies from PHAs using a Linear model"""
        # beta = np.array([0.00029353, 0.06787126]) #Huber Loss function
        beta_ls = np.array([0.00029483, 0.05735115])
        return pha * beta_ls[0] + beta_ls[1]


class simulated(builder):
    def __init__(self, total, datasets):
        super().__init__(total)
        try:
            # Create results Directory
            os.makedirs(os.path.join(self.out_base,'train/'))
            os.makedirs(os.path.join(self.out_base,'val/'))
            os.makedirs(os.path.join(self.out_base,'test/'))
            print("Directory Created for Simulated data") 
        except FileExistsError:
            print("Directory already exists for Simulated")

        self.out_files = [os.path.join(self.out_base,'train/'),  
                            os.path.join(self.out_base,'val/'), 
                            os.path.join(self.out_base,'test/')] 
        
        assert isinstance(datasets, list)
        self.datasets = datasets

        self.build_result.update([("angles", torch.zeros( (self.total, self.augment) )), 
                                    ("abs", torch.zeros( (self.total, self.augment, self.shift, 2) )),
                                     ("energy", torch.zeros( self.total, self.augment )), ("z", torch.zeros( self.total ))])

    def build(self,):
        cur_idx = 0
        for dataset in self.datasets:
            n_train_final = dataset.total
            if os.path.isdir(self.in_file):
                print("Building ", dataset.total, "tracks of", dataset.file)
                fits_data = super().init_build(os.path.join(self.in_file, dataset.file),)
            else:
                print("Building ", dataset.total, "tracks of", self.in_file)
                fits_data = super().init_build(self.in_file)

            cut = np.ones(len(fits_data[1]['PE_PHI']), dtype=bool)

            #Only take tracks in the peaks
            if self.peak_only:
                cut *= (fits_data[1]['ABS_Z'] >= 0.835) * (fits_data[1]['ABS_Z'] <= 10.83)
            if self.tailvpeak is not None:
                cut = (fits_data[1]['ABS_Z'] < 0.835) + (fits_data[1]['ABS_Z'] > 10.83)
                cut += (fits_data[1]['ABS_Z'] >= 0.835) * (fits_data[1]['ABS_Z'] <= 10.83) * np.random.choice(2, len(fits_data[1]['ABS_Z']), 
                                                                                                        p=[1-self.tailvpeak, self.tailvpeak],).astype('bool')

            print(len(fits_data[0]['DETPHI']),"loaded ok")
            print(len(fits_data[0]['DETPHI'][cut]), "post cut")
            assert len(fits_data[0]['DETPHI'][cut]) >= n_train_final, "Too few tracks {}, N_final {} too large.".format(len(fits_data[0]['DETPHI'][cut]), n_train_final)

            tracks, angles, mom_phis, abs_pts, mom_abs_pts, flags = hex2square(fits_data, cut=cut, n_final=n_train_final, augment=self.augment)

            self.build_result["tracks"].extend(tracks)
            self.build_result["angles"][cur_idx : (cur_idx + n_train_final)] = angles
            self.build_result["mom_phis"][cur_idx : (cur_idx + n_train_final)] = mom_phis
            self.build_result["moms"][cur_idx : (cur_idx + n_train_final)] = torch.from_numpy(fits_data[0]['TRK_M2L'][cut][:n_train_final] / fits_data[0]['TRK_M2T'][cut][:n_train_final]).repeat(self.augment,1).T
            self.build_result["mom_energy"][cur_idx : (cur_idx + n_train_final)] = torch.from_numpy(super().predict_energy(fits_data[0]['PHA'][cut][:n_train_final]))
            self.build_result["abs"][cur_idx : (cur_idx + n_train_final)] = abs_pts 
            self.build_result["mom_abs"][cur_idx : (cur_idx + n_train_final)] = mom_abs_pts
            self.build_result["energy"][cur_idx : (cur_idx + n_train_final)] = torch.from_numpy(fits_data[1]['ENERGY'][cut][:n_train_final].astype(np.float32)).repeat(self.augment,1).T
            self.build_result["xy_abs"][cur_idx : (cur_idx + n_train_final)] = torch.from_numpy(np.column_stack((fits_data[0]['DETX'][cut][:n_train_final],fits_data[0]['DETY'][cut][:n_train_final])))
            self.build_result["z"][cur_idx : (cur_idx + n_train_final)] = torch.from_numpy(fits_data[1]['ABS_Z'][cut][:n_train_final].astype(np.float32))
            self.build_result["flag"][cur_idx : (cur_idx + n_train_final)] = flags

            cur_idx += n_train_final


    def save(self, split, split_func):
        train_inds = super().save(split, split_func)
        if (self.augment == 1): #for training set
            train_meanE = torch.mean(self.build_result["energy"][train_inds], dim=0)
            train_stdE = torch.std(self.build_result["energy"][train_inds], dim=0)
            tracks_cum_train = [torch.from_numpy(self.build_result["tracks"][idx]) for idx in train_inds]
            train_mean, train_std = sparse_mean(tracks_cum_train, self.n_pixels)

            torch.save( (train_mean, train_std), os.path.join(self.out_base,'train/ZN.pt'))
            torch.save( (train_meanE, train_stdE), os.path.join(self.out_base,'train/ZNE.pt'))


class measured(builder):
    def __init__(self, total):
        super().__init__(total)
        try:
            # Create results Directory
            os.makedirs(self.out_base)
            print("Directory Created ") 
        except FileExistsError:
            print("Directory already exists")
        self.out_files = [self.out_base]

        self.build_result.update([("trg_id", torch.zeros( self.total, dtype=torch.int32 ))])

    def build(self,):
        fits_data = super().init_build(self.in_file,)

        print(len(fits_data['DETPHI']),"loaded ok")

        tracks, mom_phis, mom_abs_pts, flags = hex2square(fits_data, augment=self.augment)

        self.build_result["tracks"].extend(tracks[:self.total])
        self.build_result["mom_phis"][:self.total] = mom_phis[:self.total].unsqueeze(1)
        self.build_result["moms"][:self.total] = torch.from_numpy(fits_data['TRK_M2L'][:self.total] / fits_data['TRK_M2T'][:self.total]).repeat(self.augment,1).T
        self.build_result["mom_energy"][:self.total] = torch.from_numpy(super().predict_energy(fits_data['PHA'][:self.total]))
        self.build_result["mom_abs"][:self.total] = mom_abs_pts[:self.total]
        self.build_result["trg_id"][:self.total] = torch.from_numpy(fits_data['TRG_ID'][:self.total].astype(np.int32))
        self.build_result["flag"][:self.total] = flags[:self.total]
        self.build_result["xy_abs"][:self.total] = torch.from_numpy(np.column_stack((fits_data['DETX'][:self.total],fits_data['DETY'][:self.total]))) #moment abs pts in hex grid to anchor square grid coordinates



def main():
    meas_split = (1, 0) #should sum to 1
    sim_split = (0.95, 0.025, 0.025)

    if not args.sim:
        #get total number of tracks
        if not args.tot:
                with fits.open(args.input_file, memmap=False) as hdu:
                    args.tot = hdu[1].header['NAXIS2']
        print("Total number of unique tracks = {}\n".format(args.tot))

        Builder = measured(args.tot)
        Builder.build()
        Builder.save(meas_split, tt_random_split)
    else:
        if os.path.isdir(args.input_file):
            Ns = [args.tot,] #0.0566quad, 0.0496peri
            suffixs = [""]
            datasets = []
            for s,N in zip(suffixs, Ns):
                datasets.append(Dataset('gen4_spec_true_flat' + s + "_recon.fits", N))
        else:
            Ns = [args.tot,]
            datasets = [Dataset("", args.tot)]

        Builder = simulated(int(sum(Ns)), datasets)
        Builder.build()
        Builder.save(sim_split, tvt_random_split)

if __name__ == '__main__':
    main()


