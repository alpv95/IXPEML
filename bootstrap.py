import sys
import os
sys.path.insert(0, '/home/groups/rwr/alpv95/tracksml')
from ipopt import minimize_ipopt
import pickle
import numpy as np
import multiprocess as mp
from matplotlib import rcParams
# Increase the default DPI, and change the file type from png to pdf 
rcParams['savefig.dpi']           = 300
#rcParams['savefig.extension']     = "pdf"

# Simplify paths by removing "invisible" points, useful for reducing
# file size when plotting a large number of points
rcParams['path.simplify']         = True

# Instead of individually increasing font sizes, point sizes, and line 
# thicknesses, I found it easier to just decrease the figure size so
# that the line weights of various components still agree 
rcParams['figure.figsize']        = 4,4

# In this example I am *not* setting "text.usetex : True", therefore the     
# following ensures that the fonts in math mode agree with the regular ones.  
# 
rcParams['font.family']           = "serif"
rcParams['mathtext.fontset']      = "custom"
rcParams['errorbar.capsize']      = 3

# Increase the tick-mark lengths (defaults are 4 and 2)
rcParams['xtick.major.size']      = 6
rcParams['ytick.major.size']      = 6 
rcParams['xtick.minor.size']      = 3   
rcParams['ytick.minor.size']      = 3

rcParams['xtick.direction']      = "in"
rcParams['ytick.direction']      = "in" 
rcParams['xtick.top']      = True
rcParams['ytick.right']      = True 

# Increase the tick-mark widths as well as the widths of lines 
# used to draw marker edges to be consistent with the other figure
# linewidths (defaults are all 0.5)
rcParams['xtick.major.width']     = 1
rcParams['ytick.major.width']     = 1
rcParams['xtick.minor.width']     = 1
rcParams['ytick.minor.width']     = 1
rcParams['lines.markeredgewidth'] = 1

# Have the legend only plot one point instead of two, turn off the 
# frame, and reduce the space between the point and the label  
rcParams['legend.numpoints']      = 1
rcParams['legend.frameon']        = False
rcParams['legend.handletextpad']      = 0.3
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LogNorm
from util.net_test import *
from astropy.io import fits
from astropy import stats
import scipy
import pandas as pd
from scipy.signal import savgol_filter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('B', type=int,
                    help='Number of boostrap samples')
parser.add_argument("pl", type=int, choices=[0,1,2], help="PL to test")
parser.add_argument("--flat", action="store_true", help="Have effective area of telescope included or not for 0pl")
parser.add_argument("--pl_draws", type=int, default=1, help="How many random PLs to draw")
parser.add_argument("--E", nargs=2, type=float, default=(1.8,8.3),help="Energy range to consider")
args = parser.parse_args()

home_dir = '/home/groups/rwr/alpv95/tracksml/'


with open(home_dir + "final0_train___bestOG__ensemble_paper.pickle", "rb") as file:
    A = pickle.load(file)

angles, angles_mom, angles_sim, moms, errors, abs_pts, mom_abs_pts, abs_pts_sim, \
energies, energies_sim, angles1, errors1 = A

N = 14

angles = np.reshape(angles, (-1,N), order="F")
angles_mom = np.reshape(angles_mom, (-1,N), order="F")
angles_sim = np.reshape(angles_sim, (-1,N), order="F")
moms = np.reshape(moms, (-1,N), order="F")
energies = np.reshape(energies, (-1,N), order="F")
energies_sim = np.reshape(energies_sim, (-1,N), order="F")
errors = np.reshape(errors, (-1,N), order="F")
angles1 = np.reshape(angles1, (-1,N), order="F")
errors1 = np.reshape(errors1, (-1,N), order="F")
abs_pts = np.reshape(abs_pts, (-1,N), order="F")
mom_abs_pts = np.reshape(mom_abs_pts, (-1,N), order="F")
abs_pts_sim = np.reshape(abs_pts_sim, (-1,N), order="F")

meanmax1 = 1.9
meanmax2 = 2.34
meanmin2 = 0.5
#OG
errors[:,(0,2,3,5,6,8,9,)] = errors[:,(0,2,3,5,6,8,9,)] * (meanmax2 - meanmin2) / (meanmax1 - meanmin2) - meanmin2* (meanmax2 - meanmin2) / (meanmax1 - meanmin2)  + meanmin2

#if args.mean:
#    angles = np.average(angles, axis=1, weights=errors**(-1))
#    print(angles.shape)
#    angles_mom = np.mean(angles_mom, axis=1)
#    angles_sim = np.mean(angles_sim, axis=1)
#    moms = np.mean(moms, axis=1)
#    energies_sim = np.mean(energies_sim, axis=1)
#    errors = np.mean(errors, axis=1)
#    # energies = np.mean(energies, axis=1)
#    # angles1 = np.mean(angles1, axis=1)
#    # errors1 = np.mean(errors1, axis=1)
#    # abs_pts = np.mean(abs_pts, axis=1)
#    # mom_abs_pts = np.mean(mom_abs_pts, axis=1)
#    # abs_pts_sim = np.mean(abs_pts_sim, axis=1)

#mask = np.ones_like(angles, dtype=bool)
# mask[:,0] = False
# mask[:,1] = False
# mask[:,2] = False
# mask[:,3] = False
# mask[:,4] = False
# mask[:,5] = False
# mask[:,6] = False
# mask[:,7] = False
# mask[:,8] = False
# mask[:,9] = False
# mask[:,10] = False
# mask[:,11] = False
# mask[:,12] = False
# mask[:,13] = False
#mask[:,14] = False
#mask[:,15] = False
#mask[:,16] = False
#mask[:,17] = False
#mask[:,18] = False
#mask[:,19] = False
#mask[:,20] = False
# mask[:,21] = False
# mask[:,23] = False
# mask[:,16:23] = False

#angles = angles[mask]
#angles_mom = angles_mom[mask]
#angles_sim = angles_sim[mask]
#energies_sim = energies_sim[mask]
#moms = moms[mask]
#errors = errors[mask]
# angles1 = angles1[mask]
# errors1 = errors1[mask]
#try:
#    energies = energies[mask]
#    abs_pts = abs_pts[mask]
#    mom_abs_pts = mom_abs_pts[mask]
#    abs_pts_sim = abs_pts_sim[mask]
#except IndexError:
#    pass


E = set(energies_sim[:,0])
E = np.sort(list(E))

net_list = ["gen4_MSERRALL2dropPL2_aug1/models/RLRP_256_151.ptmodel"]
data_list = ["gen4_paper2/train/"]
t = NetTest(nets=net_list, fitmethod="stokes", datasets=data_list, n_nets=14)



###########################################################################
#For PLs

Aeff= [49.8,62.61,68.92,81.09, 83.8,90.74,93.69,95.11, 
       92.9,90.68,89.56,87.46,83.67,79.43, 74.2,72.45,69.46, 
       65.7,62.69,59.65,56.27,52.82,49.53,46.54,44.29,42.03,39.61,
       37.12,34.98,33.04,31.06,29.64,28.28,26.96,25.68,24.39,   
       23,21.62,20.45,19.49,18.54,17.65,16.75,15.88,15.19,14.46,13.74,
       13.27,12.81,12.12,11.07,10.12,9.243,8.403,7.758,7.132,6.521,5.937,
       5.198,4.458,3.657,3.355,3.114,2.822,2.162,]

Aeff = Aeff / np.max(Aeff)

if args.pl == 1:
    norm_factor = 1.552
elif args.pl == 2:
    norm_factor = 2.274
else:
    norm_factor = 1.0

mu_samples = []
muW025_samples = []
muW05_samples = []
muW075_samples = []
muW1_samples = []
muW125_samples = []
muW15_samples = []
muW175_samples = []
muW2_samples = []

chunks = []
for _ in range(args.pl_draws):
    # muW1_err_bootstrap = []
    # muW_err_bootstrap = []
    # mu_err_bootstrap = []
    anglesE = []
    angles_momE = []
    momsE = []
    errorsE = [] 
    energies_simE = []
    angles_simE = []

    for i,e in enumerate(E[:]):
        if e * (8.2 - 1.8) + 1.8 <= args.E[1] and e * (8.2 - 1.8) + 1.8 >= args.E[0]:
            print(e * (8.2 - 1.8) + 1.8)
            cut = (angles_sim[:,0] != 0) * (e == energies_sim[:,0])
            if args.flat:
                pl_factor = int(np.sum(cut) * (e * (8.2 - 1.8) + 1.8)**(-args.pl) * 0.95)
            else:
                pl_factor = int(np.sum(cut) * (e * (8.2 - 1.8) + 1.8)**(-args.pl) * norm_factor**2  * Aeff[i])
            #pl_factor = int(np.sum(cut))
            idx = np.random.choice(np.arange(np.sum(cut)), pl_factor, replace=False)
            anglesE.append(angles[cut,:][idx,:])
            angles_momE.append(angles_mom[:,0][cut][idx]) 
            momsE.append(moms[:,0][cut][idx]) 
            errorsE.append(errors[cut,:][idx,:] )
            energies_simE.append(energies_sim[:,0][cut][idx])
            angles_simE.append(angles_sim[:,0][cut][idx])
        
    print("Total number of tracks >> ",np.concatenate(anglesE).ravel().shape[0] / 14)

    n_cpu = os.cpu_count()
    print("Beginning parallelization on {} cores\n".format(n_cpu))
    chunks += [np.random.choice(np.arange(len(np.concatenate(angles_momE).ravel())),len(np.concatenate(angles_momE).ravel()), replace=True) for _ in range(args.B)]

    def sub(idxs):
        A1 = (np.concatenate(anglesE)[idxs].ravel(), np.concatenate(angles_momE).ravel()[idxs], np.concatenate(angles_simE).ravel()[idxs],
        np.concatenate(momsE).ravel()[idxs], np.concatenate(errorsE)[idxs].ravel(), [None], [None], [None], [None], 
        np.concatenate(energies_simE).ravel()[idxs],angles1,errors1,[None])
        mu1, _, _, _ = t.fit_mod(A1, method='stokes')
        mu2, _, _, _ = t.fit_mod(A1, method='weighted_MLE', error_weight=0.25)
        mu3, _, _, _ = t.fit_mod(A1, method='weighted_MLE', error_weight=0.5)
        mu4, _, _, _ = t.fit_mod(A1, method='weighted_MLE', error_weight=0.75)
        mu5, _, _, _ = t.fit_mod(A1, method='weighted_MLE', error_weight=1)
        mu6, _, _, _ = t.fit_mod(A1, method='weighted_MLE', error_weight=1.25)
        mu7, _, _, _ = t.fit_mod(A1, method='weighted_MLE', error_weight=1.5)
        mu8, _, _, _ = t.fit_mod(A1, method='weighted_MLE', error_weight=1.75)
        mu9, _, _, _ = t.fit_mod(A1, method='weighted_MLE', error_weight=2)
        return mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8, mu9 

    with mp.Pool(processes=n_cpu) as pool:
        results = pool.map(sub, chunks)
    print("DONE!")
    mu_err_bootstrap, muW025_err_bootstrap, muW05_err_bootstrap, muW075_err_bootstrap, muW1_err_bootstrap, muW125_err_bootstrap, muW15_err_bootstrap, muW175_err_bootstrap, muW2_err_bootstrap = zip(*results)
    mu_samples += mu_err_bootstrap
    muW025_samples += muW025_err_bootstrap
    muW05_samples += muW05_err_bootstrap
    muW075_samples += muW075_err_bootstrap
    muW1_samples += muW1_err_bootstrap
    muW125_samples += muW125_err_bootstrap
    muW15_samples += muW15_err_bootstrap
    muW175_samples += muW175_err_bootstrap
    muW2_samples += muW2_err_bootstrap

if args.pl_draws:
    np.save("bootstrapSPL{}_draws{}_dist_flat{}_range{}_{}".format(args.pl, args.pl_draws, int(args.flat), args.E[0], args.E[1]), (mu_samples, muW025_samples, muW05_samples, muW075_samples, 
                                                                                                                                    muW1_samples, muW125_samples, muW15_samples, muW175_samples, muW2_samples) )
else:
    np.save("bootstrapPL{}_dist".format(args.pl), (muW1_samples, muW_samples, mu_samples) )






#############################################################################

#muW_err_bootstrap = np.zeros((8,8,M)) 


# mu_list = []
# phi_list = []
# mu_err_list = []
# phi_err_list = []

# muW_list = []
# muW1_list = []
# muW3_list = []
# phiW_list = []
# muW_err_list = []
# muW1_err_list = []
# muW3_err_list = []
# phiW_err_list = []


# abs_acc = []
# abs_acc_err = []
# abs_acc_round = []
# abs_mom_acc = []
# abs_mom_acc_err = []
# energy_acc = []

# it = iter(E)
# ee = list(zip(it,it))
# N = []
# for j, (e1, e2) in enumerate([(E[1],E[6]),(E[6],E[11]),(E[21],E[26]),(E[31],E[36]),(E[36],E[41]), (E[41],E[46]) ,(E[51],E[56]), (E[61],E[64])]): #[(E[1],E[6]),(E[6],E[11]),(E[11],E[16]),(E[16],E[21]),(E[21],E[26]),
#               #(E[26],E[31]),(E[31],E[36]),(E[36],E[41]), (E[41],E[46]),
#             #(E[46],E[51]),(E[51],E[56]),(E[56],E[61]),(E[61],E[64])]:
#     print(e1 * (8.2 - 1.8 ) + 1.8 )
# #     cut = (data['NUM_CLU'] > 0)*(abs(data['BARX']) < 6.3)*(abs(data['BARY']) < 6.3)
# #     med = np.median(data['PI'])
# #     dev = stats.median_absolute_deviation(data['PI'])
# #     cut *= (data['PI'] > med - 3*dev) * (data['PI'] < med + 3*dev)
#     cut = (angles_sim != 0) * (e1 <= energies_sim) * (e2 >= energies_sim)
#     #pl_factor = int(np.sum(cut) * (e * (8.3 - 1.8 + 1) + 1.8 - 0.5)**(-2) * 1.75**2 )
#     anglesE = angles[cut]
#     angles_momE = angles_mom[cut] 
#     momsE = moms[cut]
#     errorsE = errors[cut] 
#     abs_ptsE = abs_pts
#     mom_abs_ptsE = mom_abs_pts
#     abs_pts_simE = abs_pts_sim
#     energiesE = energies
#     energies_simE = energies_sim[cut]
#     angles_simE = angles_sim[cut]
#     angles1E = angles1
#     errors1E = errors1
#     N.append(len(anglesE))
    
# #     abs_acc_hist = np.sum((abs_ptsE*30 - abs_pts_simE*30)**2,axis=1)
# #     abs_mom_acc_hist = np.sum((mom_abs_ptsE*30 - abs_pts_simE*30)**2,axis=1)
    
# #     abs_acc.append(np.sum((abs_ptsE*30 - abs_pts_simE*30)**2) / len(abs_ptsE))
# #     abs_acc_round.append(np.sum((np.round(abs_ptsE*30) - abs_pts_simE*30)**2) / len(abs_ptsE))
# #     abs_mom_acc.append(np.sum((mom_abs_ptsE*30 - abs_pts_simE*30)**2) / len(mom_abs_ptsE))
# #     energy_acc.append(np.mean(((energiesE - energies_simE)*((8.2 - 1.8 ) + 1.8))**2))
# #     mu_temp = []
# #     phi0_temp = []
#     print("Size: ", len(anglesE))

#     for i in range(args.B):
#         idxs = np.random.choice(np.arange(len(anglesE)),int(len(anglesE)))
#         A1 = (anglesE[idxs], angles_momE[idxs], angles_simE[idxs], momsE[idxs], errorsE[idxs], abs_ptsE, mom_abs_ptsE, 
#               abs_pts_simE, energiesE, energies_simE, angles1E, errors1E)
        
#         #plt.hist(angles_mom1,density=True,alpha=0.3,bins=50)
#         mu, phi0, mu_err, phi0_err = t.fit_mod(A1, method='stokes')
#         print(mu[0],mu[1],mu[2])
#         print(phi0[0],phi0[1],phi0[2])
#         mu_list.append(mu)
#         phi_list.append(phi0)
#         mu_err_list.append(mu_err)
#         phi_err_list.append(phi0_err)
        
#         mu0, phi00, mu_err0, phi0_err0 = t.fit_mod(A1, method='weighted_MLE', error_weight=0)
#         mu05, phi005, mu_err05, phi05_err0 = t.fit_mod(A1, method='weighted_MLE', error_weight=0.5)
#         mu2, phi02, mu_err2, phi0_err2 = t.fit_mod(A1, method='weighted_MLE', error_weight=2)
#         mu1, phi01, mu_err1, phi0_err1 = t.fit_mod(A1, method='weighted_MLE', error_weight=1)
#         mu3, phi03, mu_err3, phi0_err3 = t.fit_mod(A1, method='weighted_MLE', error_weight=3)
#         mu4, phi04, mu_err4, phi0_err4 = t.fit_mod(A1, method='weighted_MLE', error_weight=4)



#         muW_err_bootstrap[j,0,i] = mu[1]
#         muW_err_bootstrap[j,1,i] = mu0[1]
#         muW_err_bootstrap[j,2,i] = mu0[0]
#         muW_err_bootstrap[j,3,i] = mu05[0]
#         muW_err_bootstrap[j,4,i] = mu1[0]
#         muW_err_bootstrap[j,5,i] = mu2[0]
#         muW_err_bootstrap[j,6,i] = mu3[0]
#         muW_err_bootstrap[j,7,i] = mu4[0]
        
# np.save("bootstrap_dist", muW_err_bootstrap)
        
#     muW_err_bootstrap.append(np.std(mu_temp))
#     phiW_err_bootstrap.append(np.std(phi0_temp))
    
#     print(muW_err_bootstrap)
#     print(phiW_err_bootstrap)
#    print("MLE:", mu[0], mu[1])
#    print("MLE:", phi0[0], phi0[1])
    
    
    
    
#     print("abs_pt:", abs_acc[-1])
#     print("abs_pt_round:", abs_acc_round[-1])
#     print("abs_mom_pt", abs_mom_acc[-1])
    
#     print("energy:", energy_acc[-1])
    
#    muW_list.append(mu)
#    muW1_list.append(mu1)
#    muW3_list.append(mu3)
#    phiW_list.append(phi0)
#    muW_err_list.append(mu_err)
#    muW1_err_list.append(mu_err1)
#    muW3_err_list.append(mu_err3)
#    phiW_err_list.append(phi0_err)
#    print("muW_list = ",muW_list)
#    print("mu_list = ",mu_list)
#    print("muW_err_list = ",muW_err_list)
#    print("mu_err_list = ",mu_err_list)
#    
#    print("muW1_list = ",muW1_list)
#    print("muW1_err_list = ",muW1_err_list)
#  
#    print("muW3_list = ", muW3_list)
#    print("muW3_err_list = ",muW3_err_list)
