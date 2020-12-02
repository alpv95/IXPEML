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
parser.add_argument('--shuffle', action='store_true',
                    help='Whether to shuffle tracks from their original fits file order')
args = parser.parse_args()


# Specify options for datasets and their input -> output properties
# The output file is constructed from each row's name+type
Dataset = namedtuple('Dataset', ['name', 'energy', 'angle', 'pol', 'type'])


# class builder(object):
#     in_base = args.input_base
#     out_base = args.out_base
#     n_pixels = args.npix
#     augment = args.augment
#     shift = args.shift
#     pl = args.pl

#     def __init__(self, datasets, total):
#         self.total = total
#         self.datasets = datasets

#         tracks_cum = []
#         mom_phis_cum = torch.zeros( (self.total, self.augment) ) 
#         moms_cum = torch.zeros( self.total, self.augment )
#         mom_abs_cum = torch.zeros( (self.total, self.augment, self.shift, 2) )
#         xy_abs_cum = torch.zeros( (self.total, 2) ) #xy location of track on detector grid
#         self.build_result = [tracks_cum, mom_phis_cum, moms_cum, mom_abs_cum, xy_abs_cum]

#     def init_build(self, input_file, pulse_cut):
#         with fits.open(input_file, memmap=False) as hdu:
#             data = hdu[1].data
        
#         #APPLY SOME INITIAL TRACK CUTS HERE AS IN REPORT
#         cut = (data['NUM_CLU'] > 0)*(abs(data['BARX']) < 6.3)*(abs(data['BARY']) < 6.3)
#         if pulse_cut:
#             (mu,sigma) = norm.fit(data['PI'])
#             cut *= (data['PI'] > mu - 3*sigma) * (data['PI'] < mu + 3*sigma)
#         return data, cut

#     def save(self, split, split_func):
#         N_final = len(self.build_result[0])
#         indxs = split_func(np.arange(N_final), fracs=split)

#         for indx, save_file in zip(indxs,files):
#             tracks_cum_save = [torch.from_numpy(self.build_result[0][idx]) for idx in indx]
#             with open(save_file + "tracks_full.pickle", "wb") as f:
#                 pickle.dump(tracks_cum_save, f)
#             torch.save( [result[indx] for result in self.build_result], save_file + "labels_full.pt" )
#             print("Saved, ", save_file )
        
#         return indxs[0]


# class simulated(builder):
#     def __init__(self):
#         super().__init__()
#         try:
#             # Create results Directory
#             os.makedirs(os.path.join(self.out_base,'train/'))
#             os.makedirs(os.path.join(self.out_base,'val/'))
#             os.makedirs(os.path.join(self.out_base,'test/'))
#             print("Directory Created for Simulated data") 
#         except FileExistsError:
#             print("Directory already exists for Simulated")

#         sq_output_file_train = os.path.join(self.out_base,'train/') 
#         sq_output_file_val = os.path.join(self.out_base,'val/') 
#         sq_output_file_test = os.path.join(self.out_base,'test/') 

#         angles_cum = torch.zeros( (self.total, self.augment) )
#         abs_cum = torch.zeros( (self.total, self.augment, self.shift, 2) )
#         energy_cum = torch.zeros( self.total, self.augment )
#         z_cum = torch.zeros( self.total, self.augment )
#         self.build_result.extend([angles_cum, abs_cum, energy_cum, z_cum])

#     def build(self, pulse_cut):
#         cur_idx = 0
#         for dataset in self.datasets:
#             print('Building dataset for {name} {energy} keV {angle} deg {type}'.format(**dataset._asdict()))
#             energy_str = '{:.1f}'.format(dataset.energy).replace('.', 'p')
#             input_file = os.path.join(self.in_base, '{}_{}{}_recon.fits'.format(dataset.name, 
#                                                                         angle_str, energy_str))
#             n_train_final = int(dataset.energy**(-self.pl) * 3070000 * fraction * aeff(dataset.energy))

#             data, cut = super().init_build(input_file, pulse_cut)
#             with fits.open(input_file.replace('_recon',''), memmap=False) as hdu:
#                 sim_data = hdu[3].data
#             cut *= (sim_data['PE_PHI'] != 0.0) #to remove bump in training data

#             moms = (data['TRK_M2L'] / data['TRK_M2T'])[cut]
#             Zs = sim_data['ABS_Z'][cut]
#             mom_abs_pts = np.column_stack((data['DETX'],data['DETY']))[cut]
#             bars = np.column_stack((data['BARX'],data['BARY']))[cut]
#             abs_pts = np.column_stack((sim_data['ABS_X'],sim_data['ABS_Y']))[cut]
#             hex_tracks = SparseHexSimTracks(data['PIX_X'][cut], data['PIX_Y'][cut], data['PIX_PHA'][cut], 
#                                             sim_data['PE_PHI'][cut], abs_pts, moms, data['DETPHI'][cut], 
#                                             mom_abs_pts, bars)
#             hex_tracks = hex_tracks[np.arange(n_train_final)]

#             tracks, angles, mom_phis, abs_pts, mom_abs_pts = hex2square(hex_tracks, self.n_pixels, augment=self.augment, shift=self.shift)

#             self.build_result[0].extend(tracks)
#             self.build_result[5][cur_idx : (cur_idx + n_train_final)] = angles 
#             self.build_result[1][cur_idx : (cur_idx + n_train_final)] = mom_phis
#             self.build_result[2][cur_idx : (cur_idx + n_train_final)] = torch.from_numpy(hex_tracks.mom).repeat(self.augment,1).T
#             self.build_result[6][cur_idx : (cur_idx + n_train_final)] = abs_pts 
#             self.build_result[3][cur_idx : (cur_idx + n_train_final)] = mom_abs_pts 
#             self.build_result[7][cur_idx : (cur_idx + n_train_final)] = torch.from_numpy( dataset.energy * np.ones_like(hex_tracks.mom)).repeat(self.augment,1).T
#             self.build_result[4][cur_idx : (cur_idx + n_train_final)] = torch.from_numpy(hex_tracks.mom_abs_pt)
#             self.build_result[8][cur_idx : (cur_idx + n_train_final)] = torch.from_numpy(Zs * np.ones_like(hex_tracks.mom)).repeat(self.augment,1).T

#             cur_idx += n_train_final


#     def save(self, split, split_func):
#         train_inds = super().save(split, split_func)

#         train_meanE = torch.mean(self.build_result[7][train_inds], dim=0)
#         train_stdE = torch.std(self.build_result[7][train_inds], dim=0)
#         train_mean, train_std = sparse_mean(self.build_result[0], self.n_pixels)
#         torch.save( (train_mean, train_std), os.path.join(self.out_base,'train/ZN.pt'))
#         torch.save( (train_meanE, train_stdE), os.path.join(self.out_base,'train/ZNE.pt'))


# class measured(builder):
#     def __init__(self):
#         super().__init__()
#         try:
#             # Create results Directory
#             os.makedirs(os.path.join(self.out_base,'meas_{}_{}_{}/train/'.format(energy_str, dataset.angle, dataset.pol)))
#             os.makedirs(os.path.join(self.out_base,'meas_{}_{}_{}/test/'.format(energy_str, dataset.angle, dataset.pol)))
#             print("Directory Created ") 
#         except FileExistsError:
#             print("Directory already exists")
        
#         sq_output_file_train = os.path.join(self.out_base,'meas_{}_{}_{}/train/'.format(energy_str, dataset.angle, dataset.pol)) 
#         sq_output_file_test = os.path.join(self.out_base,'meas_{}_{}_{}/test/'.format(energy_str, dataset.angle, dataset.pol))

#     def build(self, meas_file, pulse_cut):
#         input_file = os.path.join(self.in_base, meas_file)
#         data, cut = super().init_build(input_file, pulse_cut)

#         moms = (data['TRK_M2L'] / data['TRK_M2T'])[cut]
#         mom_abs_pts = np.column_stack((data['DETX'],data['DETY']))[cut]
#         bars = np.column_stack((data['BARX'],data['BARY']))[cut]
#         hex_tracks = SparseHexTracks(data['PIX_X'][cut], data['PIX_Y'][cut], data['PIX_PHA'][cut], 
#                                          moms, data['DETPHI'][cut], mom_abs_pts, bars) 

#         hex_tracks = hex_tracks[np.arange(self.total)]
                    
#         tracks, mom_phis, mom_abs_pts = hex2square(hex_tracks, self.n_pixels, augment=self.augment, shift=self.shift)

#         self.build_result[0].extend(tracks)
#         self.build_result[1][:self.total] = mom_phis.unsqueeze(1)
#         self.build_result[2][:self.total] = torch.from_numpy(hex_tracks.mom).repeat(self.augment,1).T
#         self.build_result[3][:self.total] = mom_abs_pts 
#         self.build_result[4][:self.total] = torch.from_numpy(hex_tracks.mom_abs_pt) #moment abs pts in hex grid to anchor square grid coordinates



# def main():
#     input_base, out_base, augment, shift, n_pixels, Erange, \
#     fraction, pl, aeff, meas, meas_tot, meas_e, pulse, shuffle = args
#     # meas dataset 2-way vt split
#     meas_split = (0.9, 0.1)
#     # sim dataset 3-way tvt split
#     sim_split = (0.9, 0.05, 0.05)


def main():
    input_base, out_base, augment, shift, n_pixels, Erange, \
    fraction, pl, aeff_on, meas, meas_tot, meas_e, pulse_cut, shuffle = args.input_base, args.out_base, args.augment, args.shift, args.npix, args.Erange, \
    args.fraction, args.pl, args.aeff, args.meas, args.meas_tot, args.meas_e, args.pulse_cut, args.shuffle
    print("out_base: ", out_base)

    # meas dataset 2-way vt split
    meas_split = (0.9, 0.1)
    # sim dataset 3-way tvt split
    sim_split = (0.9, 0.05, 0.05)

    if aeff_on:
        aeff = lambda E: Aeff(E)
    else:
        aeff = lambda E: 1

    datasets = []
    energies = np.round(np.arange(Erange[0], Erange[1], 0.1),2)
    angles = []#np.linspace(0,170,18).astype('int')
    for energy in energies:
        datasets.append(Dataset('gen4', round(energy, 1), None, 'pol', 'sim'))
    for angle in angles:
        for energy in energies:
            datasets.append(Dataset('gen4', round(energy, 1), angle, 'pol', 'sim'))
    #datasets = [Dataset('gen4', round(6.4, 1), 120, 'pol', 'sim'),Dataset('gen4', round(6.5, 1), 120, 'pol', 'sim')]
    if meas:
        datasets = [Dataset('gen4', meas_e, None, 'pol', 'meas')]

    if not os.path.exists(out_base):
        os.makedirs(out_base)

    if not meas:
        try:
            # Create results Directory
            os.makedirs(os.path.join(out_base,'train/'))
            os.makedirs(os.path.join(out_base,'val/'))
            os.makedirs(os.path.join(out_base,'test/'))
            print("Directory Created for Simulated data") 
        except FileExistsError:
            print("Directory already exists for Simulated")
            
        total = 0
        for energy in energies:
            total += int(energy**(-pl) * 3070000 * fraction * aeff(energy) ) + len(angles) * int( energy**(-pl) * 380000 * fraction * aeff(energy) )
        print("Total number of unique tracks = {}\n".format(total))
        angles_cum = torch.zeros( (total, augment) )
        tracks_cum = []
        mom_phis_cum = torch.zeros( (total, augment) ); moms_cum = torch.zeros( total, augment )
        abs_cum = torch.zeros( (total, augment, shift, 2) )
        mom_abs_cum = torch.zeros( (total, augment, shift, 2) )
        energy_cum = torch.zeros( total, augment )
        xy_abs_cum = torch.zeros( (total, 2) ) 
    else:
        total = meas_tot
        print("Total number of unique tracks to square = {}\n".format(total))
        tracks_cum = []
        mom_phis_cum = torch.zeros( (total, augment) ); moms_cum = torch.zeros( total, augment )
        mom_abs_cum = torch.zeros( (total, augment, shift, 2) )
        xy_abs_cum = torch.zeros( (total, 2) ) #xy location of track on detector grid
    cur_idx = 0
        


    for dataset in datasets:
        print('Building dataset for {name} {energy} keV {angle} deg {type}'.format(**dataset._asdict()))
        energy_str = '{:.1f}'.format(dataset.energy).replace('.', 'p')
        if dataset.type == 'sim':
           if dataset.angle is not None:
               angle_str = 'ang{}_'.format(dataset.angle)
           else:
               angle_str = ''
           input_file = os.path.join(input_base, '{}_{}{}_recon.fits'.format(
            dataset.name, angle_str, energy_str
           ))
        else: 
           input_file = os.path.join(input_base, meas)

        with fits.open(input_file, memmap=False) as hdu:
            data = hdu[1].data
        
        #APPLY SOME INITIAL TRACK CUTS HERE AS IN REPORT
        cut = (abs(data['BARX']) < 6.3)*(abs(data['BARY']) < 6.3) * (data['NUM_CLU'] > 0)
        if pulse_cut:
            (mu,sigma) = norm.fit(data['PI'])
            cut *= (data['PI'] > mu - 3*sigma) * (data['PI'] < mu + 3*sigma)

        if dataset.type == 'sim':
            with fits.open(input_file.replace('_recon',''), memmap=False) as hdu:
                sim_data = hdu[3].data
            cut *= (sim_data['PE_PHI'] != 0.0) #to remove bump in training data

            moms = (data['TRK_M2L'] / data['TRK_M2T'])[cut]
            mom_abs_pts = np.column_stack((data['DETX'],data['DETY']))[cut]
            bars = np.column_stack((data['BARX'],data['BARY']))[cut]
            abs_pts = np.column_stack((sim_data['ABS_X'],sim_data['ABS_Y']))[cut]
            hex_tracks = SparseHexSimTracks(data['PIX_X'][cut], data['PIX_Y'][cut], data['PIX_PHA'][cut], sim_data['PE_PHI'][cut], abs_pts, moms, data['DETPHI'][cut], mom_abs_pts, bars) 
        else:
            moms = (data['TRK_M2L'] / data['TRK_M2T'])[cut]
            mom_abs_pts = np.column_stack((data['DETX'],data['DETY']))[cut]
            bars = np.column_stack((data['BARX'],data['BARY']))[cut]
            hex_tracks = SparseHexTracks(data['PIX_X'][cut], data['PIX_Y'][cut], data['PIX_PHA'][cut], moms, data['DETPHI'][cut], mom_abs_pts, bars) 

        n_tracks = hex_tracks.n_tracks
        print(n_tracks,"loaded ok")

        if dataset.type == 'sim':
            #more low energy tracks than high energy
            if dataset.angle is not None:
                n_train_final = int(dataset.energy**(-pl) * 380000 * fraction * aeff(dataset.energy))#1250 #int(4000 - dataset.energy*400)  * Aeff_train(dataset.energy)
            else:
                n_train_final = int(dataset.energy**(-pl) * 3070000 * fraction * aeff(dataset.energy)) 
            
            if shuffle:
                #randomly select subset
                hex_tracks = expand_tracks(hex_tracks, n_train_final)
            else:
                hex_tracks = hex_tracks[np.arange(n_train_final)]

            tot_tracks = hex_tracks.n_tracks
            sq_output_file_train = os.path.join(out_base,'train/') 
            sq_output_file_val = os.path.join(out_base,'val/') 
            sq_output_file_test = os.path.join(out_base,'test/') 
            
            tracks, angles, mom_phis, abs_pts, mom_abs_pts = hex2square(hex_tracks, n_pixels, augment=augment, shift=shift)

            tracks_cum.extend(tracks)
            angles_cum[cur_idx : (cur_idx + n_train_final)] = angles 
            mom_phis_cum[cur_idx : (cur_idx + n_train_final)] = mom_phis
            moms_cum[cur_idx : (cur_idx + n_train_final)] = torch.from_numpy(hex_tracks.mom).repeat(augment,1).T
            abs_cum[cur_idx : (cur_idx + n_train_final)] = abs_pts 
            mom_abs_cum[cur_idx : (cur_idx + n_train_final)] = mom_abs_pts 
            energy_cum[cur_idx : (cur_idx + n_train_final)] = torch.from_numpy( dataset.energy * np.ones_like(hex_tracks.mom)).repeat(augment,1).T
            xy_abs_cum[cur_idx : (cur_idx + n_train_final)] = torch.from_numpy(hex_tracks.mom_abs_pt)

            cur_idx += n_train_final

            print(len(tracks_cum),cur_idx)
            print(tracks_cum[cur_idx-1].shape)
            print(angles_cum.shape)



        elif dataset.type == 'meas':
            # Split 2 ways
            if shuffle:
                #randomly select subset
                hex_tracks = expand_tracks(hex_tracks, total)
            else:
                hex_tracks = hex_tracks[np.arange(total)]

            hex_tracks = expand_tracks(hex_tracks, total)
            tot_tracks = hex_tracks.n_tracks

            try:
                os.makedirs(os.path.join(out_base,'meas_{}_{}_{}/train/'.format(energy_str, dataset.angle, dataset.pol)))
                os.makedirs(os.path.join(out_base,'meas_{}_{}_{}/test/'.format(energy_str, dataset.angle, dataset.pol)))
                print("Directory Created ") 
            except FileExistsError:
                print("Directory already exists")

            sq_output_file_train = os.path.join(out_base,'meas_{}_{}_{}/train/'.format(energy_str, dataset.angle, dataset.pol)) 
            sq_output_file_test = os.path.join(out_base,'meas_{}_{}_{}/test/'.format(energy_str, dataset.angle, dataset.pol)) 
            
            tracks, mom_phis, mom_abs_pts = hex2square(hex_tracks, n_pixels, augment=augment, shift=shift)

            tracks_cum.extend(tracks)
            mom_phis_cum[cur_idx : (cur_idx + total)] = mom_phis.unsqueeze(1)
            moms_cum[cur_idx : (cur_idx + total)] = torch.from_numpy(hex_tracks.mom).repeat(augment,1).T
            mom_abs_cum[cur_idx : (cur_idx + total)] = mom_abs_pts 
            xy_abs_cum[cur_idx : (cur_idx + total)] = torch.from_numpy(hex_tracks.mom_abs_pt) #moment abs pts in hex grid to anchor square grid coordinates

            cur_idx += total

        else:
            raise ValueError('dataset type not recognized')
    
    #Save data in torch pt file
    if dataset.type == 'sim':
        N_final = len(tracks_cum) #tracks_cum.shape[0]
        if shuffle:
            print("Shuffling") #Note: NEED shuffling for unpolarized (simulated) dataset
            train_inds, val_inds, test_inds = tvt_random_split(np.arange(N_final), fracs=sim_split)
        else:
            print("Not shuffling")
            train_inds = np.arange(N_final)[:int(sim_split[0] * N_final)]
            val_inds = np.arange(N_final)[int(sim_split[0] * N_final):int((sim_split[0]+sim_split[1]) * N_final)]
            test_inds = np.arange(N_final)[int((sim_split[0]+sim_split[1]) * N_final):]

        tracks_cum_train = [torch.from_numpy(tracks_cum[idx]) for idx in train_inds]
        tracks_cum_val = [torch.from_numpy(tracks_cum[idx]) for idx in val_inds]
        tracks_cum_test = [torch.from_numpy(tracks_cum[idx]) for idx in test_inds]

        train_meanE = torch.mean(energy_cum[train_inds], dim=0)
        train_stdE = torch.std(energy_cum[train_inds], dim=0)
        train_mean, train_std = sparse_mean(tracks_cum_train, n_pixels)
        torch.save( (train_mean, train_std), sq_output_file_train + "ZN.pt")
        torch.save( (train_meanE, train_stdE), sq_output_file_train + "ZNE.pt")

        with open(sq_output_file_train + "tracks_full.pickle", "wb") as f:
            pickle.dump(tracks_cum_train, f)
        with open(sq_output_file_val + "tracks_full.pickle", "wb") as f:
            pickle.dump(tracks_cum_val, f)
        with open(sq_output_file_test + "tracks_full.pickle", "wb") as f:
            pickle.dump(tracks_cum_test, f)

        torch.save( (angles_cum[train_inds], moms_cum[train_inds], mom_phis_cum[train_inds], 
                abs_cum[train_inds], mom_abs_cum[train_inds], energy_cum[train_inds], xy_abs_cum[train_inds]), sq_output_file_train + "labels_full.pt" )
        torch.save( (angles_cum[val_inds], moms_cum[val_inds], mom_phis_cum[val_inds], 
                abs_cum[val_inds], mom_abs_cum[val_inds], energy_cum[val_inds], xy_abs_cum[val_inds]), sq_output_file_val + "labels_full.pt" )
        torch.save( (angles_cum[test_inds], moms_cum[test_inds], mom_phis_cum[test_inds], 
                abs_cum[test_inds], mom_abs_cum[test_inds], energy_cum[test_inds], xy_abs_cum[test_inds]), sq_output_file_test + "labels_full.pt" )


    else:
        N_final = len(tracks_cum)
        train_inds, test_inds = tt_random_split(np.arange(N_final), fracs=meas_split)

        tracks_cum_train = [torch.from_numpy(tracks_cum[idx]) for idx in train_inds]
        tracks_cum_test = [torch.from_numpy(tracks_cum[idx]) for idx in test_inds]

        with open(sq_output_file_train + "tracks_full.pickle", "wb") as f:
            pickle.dump(tracks_cum_train, f)
        with open(sq_output_file_test + "tracks_full.pickle", "wb") as f:
            pickle.dump(tracks_cum_test, f)

        torch.save( (moms_cum[train_inds], mom_phis_cum[train_inds], mom_abs_pts[train_inds], 
            torch.from_numpy(dataset.energy*np.ones_like(moms_cum))[train_inds], xy_abs_cum[train_inds] ), sq_output_file_train + "labels_full.pt" )
        torch.save( (moms_cum[test_inds], mom_phis_cum[test_inds], mom_abs_pts[test_inds],
            torch.from_numpy( dataset.energy*np.ones_like(moms_cum))[test_inds], xy_abs_cum[test_inds] ), sq_output_file_test + "labels_full.pt" )
    print("Saved, ", sq_output_file_train, sq_output_file_test )


if __name__ == '__main__':
    main()


