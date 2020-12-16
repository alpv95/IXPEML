'''
Takes fits files of GPDSW reconstructed hexagonal photoelectron tracks and assembles them into square tracks useable by NN ensembles.
For both real (measured) and simulated tracks. 
'''

import os
import h5py
import numpy as np
from collections import namedtuple
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from formats.sparse_hex import SparseHexTracks, SparseHexSimTracks
from prep.augment_tracks import expand_tracks
from prep.convert_hex_to_square_tracks import hex2square
from util.split_data import tvt_random_split, tt_random_split
import torch
from scipy.stats import norm
from astropy.io import fits
import argparse
from astropy import stats
import pickle
from util.methods import *

parser = argparse.ArgumentParser()
parser.add_argument('input_base', type=str,
                    help='input folder')
parser.add_argument('out_base', type=str,
                    help='output folder')
parser.add_argument('--augment', type=int, choices=[1,2,3,4,5,6], default=3,
                    help='Number of Track augmentation randomly distributed in dataset. These should always be 3 during inference for >= v1.1')
parser.add_argument('--shift', type=int, choices=[1,2], default=2,
                    help='Number of Track shifts randomly distributed in dataset. This should always be 2 for >= v1.1')
parser.add_argument('--npix', type=int, choices=[30, 50], default=50,
                    help='Number of pixels in square conversions. This should be 50 for >= v1.2')
parser.add_argument('--Erange', nargs=2, type=float, default=(1.8,8.3),
                    help='Energy Range for simulated tracks, can be up to 1 - 9keV v1.2')
parser.add_argument('--fraction', type=float, default=0.0814,
                    help='Dictates how many tracks in the given energy range will be produced, must be >0')  
parser.add_argument('--pl', type=float, default=0,
                    help='Power Law index for track Energy distribution') 
parser.add_argument('--aeff', action='store_true',
                    help='Whether to include telescope effective area for simulated tracks.')   
parser.add_argument('--meas', type=str,
                    help='Filename of measured data: Whether data is real or simulated, if squaring measured data this argument is required')
parser.add_argument('--meas_tot', type=int,
                    help='The total number of measured tracks to convert to square')
parser.add_argument('--meas_e', type=float,
                    help='Energy of measured tracks in kev')
parser.add_argument('-pulse','--pulse_cut', action='store_true',
                    help='Whether to cut data about the maximum in the pulse height spectrum as in calibration documents')
args = parser.parse_args()


# Specify options for datasets and their input -> output properties
# The output file is constructed from each row's name+type
Dataset = namedtuple('Dataset', ['name', 'energy', 'angle', 'pol', 'type'])


class builder(object):
    in_base = args.input_base
    out_base = args.out_base
    n_pixels = args.npix
    augment = args.augment
    shift = args.shift
    pl = args.pl
    fraction = args.fraction

    def __init__(self, datasets, total):
        self.total = total
        self.datasets = datasets
        self.out_files = []

        self.build_result = {"tracks": [], "mom_phis": torch.zeros( (self.total, self.augment) ), "moms": torch.zeros( self.total, self.augment ),
                             "mom_abs": torch.zeros( (self.total, self.augment, self.shift, 2) ), "xy_abs": torch.zeros( (self.total, 2) ), 
                             "mom_energy": torch.zeros( self.total)}

    def init_build(self, input_file, pulse_cut):
        with fits.open(input_file, memmap=False) as hdu:
            data = hdu[1].data

        #APPLY SOME INITIAL TRACK CUTS HERE AS IN REPORT
        cut = (data['NUM_CLU'] > 0)*(abs(data['BARX']) < 6.3)*(abs(data['BARY']) < 6.3)
        if pulse_cut:
            (mu,sigma) = norm.fit(data['PI'])
            cut *= (data['PI'] > mu - 3*sigma) * (data['PI'] < mu + 3*sigma)
        return data, cut

    def save(self, split, split_func):
        N_final = len(self.build_result["tracks"])
        indxs = split_func(np.arange(N_final), fracs=split)

        for indx, save_file in zip(indxs, self.out_files):
            tracks_cum_save = [torch.from_numpy(self.build_result["tracks"][idx]) for idx in indx]
            with open(save_file + "tracks_full.pickle", "wb") as f:
                pickle.dump(tracks_cum_save, f)
            torch.save( {key: value[indx] for (key,value) in self.build_result.items() if key != "tracks"}, save_file + "labels_full.pt" )
            print("Saved, ", save_file )
        
        return indxs[0]

    def predict_energy(self, pha):
        """predicts track energies from PHAs using a Linear model"""
        # beta = np.array([0.00029353, 0.06787126]) #Huber Loss function
        beta_ls = np.array([0.00029483, 0.05735115])
        return pha * beta_ls[0] + beta_ls[1]


class simulated(builder):
    def __init__(self, datasets, total):
        super().__init__(datasets, total)
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

        self.build_result.update([("angles", torch.zeros( (self.total, self.augment) )), 
                                    ("abs", torch.zeros( (self.total, self.augment, self.shift, 2) )),
                                     ("energy", torch.zeros( self.total, self.augment )), ("z", torch.zeros( self.total ))])

    def build(self, pulse_cut, aeff):
        cur_idx = 0
        for dataset in self.datasets:
            print('Building dataset for {name} {energy} keV {angle} deg {type}'.format(**dataset._asdict()))
            energy_str = '{:.1f}'.format(dataset.energy).replace('.', 'p')
            input_file = os.path.join(self.in_base, '{}_{}_recon.fits'.format(dataset.name, energy_str))
            n_train_final = int(dataset.energy**(-self.pl) * 3070000 * self.fraction * aeff(dataset.energy))

            data, cut = super().init_build(input_file, pulse_cut)
            with fits.open(input_file.replace('_recon',''), memmap=False) as hdu:
                sim_data = hdu[3].data

            cut *= (sim_data['PE_PHI'] != 0.0) #to remove bump in training data

            moms = (data['TRK_M2L'] / data['TRK_M2T'])[cut]
            Zs = sim_data['ABS_Z'][cut].astype(np.float32)
            mom_energies = super().predict_energy(data['PHA'][cut])
            mom_abs_pts = np.column_stack((data['DETX'],data['DETY']))[cut]
            bars = np.column_stack((data['BARX'],data['BARY']))[cut]
            abs_pts = np.column_stack((sim_data['ABS_X'],sim_data['ABS_Y']))[cut]
            hex_tracks = SparseHexSimTracks(data['PIX_X'][cut], data['PIX_Y'][cut], data['PIX_PHA'][cut], 
                                            sim_data['PE_PHI'][cut], abs_pts, moms, mom_energies, data['DETPHI'][cut], 
                                            mom_abs_pts, bars, Zs)

            print(hex_tracks.n_tracks,"loaded ok")
            hex_tracks = hex_tracks[np.arange(n_train_final)]

            tracks, angles, mom_phis, abs_pts, mom_abs_pts = hex2square(hex_tracks, self.n_pixels, augment=self.augment, shift=self.shift)

            self.build_result["tracks"].extend(tracks)
            self.build_result["angles"][cur_idx : (cur_idx + n_train_final)] = angles 
            self.build_result["mom_phis"][cur_idx : (cur_idx + n_train_final)] = mom_phis
            self.build_result["moms"][cur_idx : (cur_idx + n_train_final)] = torch.from_numpy(hex_tracks.mom).repeat(self.augment,1).T
            self.build_result["mom_energy"][cur_idx : (cur_idx + n_train_final)] = torch.from_numpy(hex_tracks.mom_energy)
            self.build_result["abs"][cur_idx : (cur_idx + n_train_final)] = abs_pts 
            self.build_result["mom_abs"][cur_idx : (cur_idx + n_train_final)] = mom_abs_pts 
            self.build_result["energy"][cur_idx : (cur_idx + n_train_final)] = torch.from_numpy( dataset.energy * np.ones_like(hex_tracks.mom)).repeat(self.augment,1).T
            self.build_result["xy_abs"][cur_idx : (cur_idx + n_train_final)] = torch.from_numpy(hex_tracks.mom_abs_pt)
            self.build_result["z"][cur_idx : (cur_idx + n_train_final)] = torch.from_numpy(hex_tracks.zs)

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
    def __init__(self, datasets, total):
        super().__init__(datasets, total)
        energy_str = '{:.1f}'.format(self.datasets[0].energy).replace('.', 'p')
        try:
            # Create results Directory
            os.makedirs(os.path.join(self.out_base,'meas_{}_{}/train/'.format(energy_str, self.datasets[0].pol)))
            os.makedirs(os.path.join(self.out_base,'meas_{}_{}/test/'.format(energy_str, self.datasets[0].pol)))
            print("Directory Created ") 
        except FileExistsError:
            print("Directory already exists")
        self.out_files = [os.path.join(self.out_base,'meas_{}_{}/train/'.format(energy_str,  self.datasets[0].pol)),  
                        os.path.join(self.out_base,'meas_{}_{}/test/'.format(energy_str,  self.datasets[0].pol))]

    def build(self, meas_file, pulse_cut):
        input_file = os.path.join(self.in_base, meas_file)
        data, cut = super().init_build(input_file, pulse_cut)

        moms = (data['TRK_M2L'] / data['TRK_M2T'])[cut]
        mom_energies = super().predict_energy(data['PHA'][cut])
        mom_abs_pts = np.column_stack((data['DETX'],data['DETY']))[cut]
        bars = np.column_stack((data['BARX'],data['BARY']))[cut]
        hex_tracks = SparseHexTracks(data['PIX_X'][cut], data['PIX_Y'][cut], data['PIX_PHA'][cut], 
                                         moms, mom_energies, data['DETPHI'][cut], mom_abs_pts, bars) 
        print(hex_tracks.n_tracks,"loaded ok")
        hex_tracks = hex_tracks[np.arange(self.total)]
                    
        tracks, mom_phis, mom_abs_pts = hex2square(hex_tracks, self.n_pixels, augment=self.augment, shift=self.shift)

        self.build_result["tracks"].extend(tracks)
        self.build_result["mom_phis"][:self.total] = mom_phis.unsqueeze(1)
        self.build_result["moms"][:self.total] = torch.from_numpy(hex_tracks.mom).repeat(self.augment,1).T
        self.build_result["mom_energy"][:self.total] = torch.from_numpy(hex_tracks.mom_energy)
        self.build_result["mom_abs"][:self.total] = mom_abs_pts 
        self.build_result["xy_abs"][:self.total] = torch.from_numpy(hex_tracks.mom_abs_pt) #moment abs pts in hex grid to anchor square grid coordinates



def main():
    meas_split = (0.9, 0.1)
    sim_split = (0.9, 0.05, 0.05)

    if args.meas:
        datasets = [Dataset('gen4', args.meas_e, None, 'pol', 'meas')]
        Builder = measured(datasets, args.meas_tot)
        Builder.build(args.meas, args.pulse_cut)
        Builder.save(meas_split, tt_random_split)
    else:
        datasets = []
        energies = np.round(np.arange(args.Erange[0], args.Erange[1], 0.1),2)
        angles = [] #np.linspace(0,170,18).astype('int')
        for energy in energies:
            datasets.append(Dataset('gen4', round(energy, 1), None, 'pol', 'sim'))

        if args.aeff:
            aeff = lambda E: paper_spec(E)
        else:
            aeff = lambda E: 1

        total = 0
        for energy in energies:
            total += int(energy**(-args.pl) * 3070000 * args.fraction * aeff(energy) ) + len(angles) * int( energy**(-args.pl) * 380000 * args.fraction * aeff(energy) )
        print("Total number of unique tracks = {}\n".format(total))
        
        Builder = simulated(datasets, total)
        Builder.build(args.pulse_cut, aeff)
        Builder.save(sim_split, tvt_random_split)

if __name__ == '__main__':
    main()


