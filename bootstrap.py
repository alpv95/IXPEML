import sys
import os
sys.path.insert(0, '/home/groups/rwr/alpv95/tracksml')
from ipopt import minimize_ipopt
import pickle
import numpy as np
import multiprocess as mp
import torch
from matplotlib.colors import LogNorm
from astropy import stats
import scipy
from util.methods import *
from scipy import interpolate
from scipy.optimize import minimize_scalar
import pandas as pd
from scipy.signal import savgol_filter
import argparse
from util.methods import *
from util.net_test import *

parser = argparse.ArgumentParser()
parser.add_argument('B', type=int,
                    help='Number of boostrap samples')
parser.add_argument("pl", type=int, choices=[0,1,2], help="PL to test")
parser.add_argument("--flat", action="store_true", help="Have effective area of telescope included or not for 0pl")
parser.add_argument("--pl_draws", type=int, default=1, help="How many random PLs to draw")
parser.add_argument("--E", nargs=2, type=float, default=(1.8,8.3),help="Energy range to consider")
args = parser.parse_args()

home_dir = '/home/groups/rwr/alpv95/tracksml/'

area = np.loadtxt("notebooks/MMA_cal-eff-area_20200831.txt")[:,(0,-1)]
eff1 = np.loadtxt("notebooks/du_efficiency_687mbar.txt")

A = (1e-11 * 6.24151e+8) / (np.log(8) - np.log(2))
N_E = (10*24*60*60) * np.interp(np.linspace(1,9,1601),area[:,0],area[:,1]) \
                         * np.interp(np.linspace(1,9,1601),eff1[:,0],eff1[:,1])/0.8 * A * np.linspace(1,9,1601)**(-args.pl) \
                         * 0.005 
f = interpolate.interp1d(np.linspace(1,9,1601), N_E, kind="cubic")

# Polarized Data
with open(home_dir + "fom_pol_big_train___flat_all_pol__ensemble.pickle", "rb") as file:
    A = pickle.load(file)
angles, angles_mom, angles_sim, moms, errors, _, _, _, _, energies_sim, _, _, _ = A

# mu1_total = []
# phi1_total = []

# mdps = []
# mu1s = []
# phi1s = []
# Neff1s = []
# for _ in range(args.pl_draws):
t = NetTest(n_nets=10)
Angles = (angles, angles_mom, angles_sim, moms, errors, energies_sim)
angles_NN_spec, angles_mom_spec, angles_sim_spec, moms_spec, errors_spec, _ = generate_spectrum(Angles, f, fraction=0.85)
print(angles_NN_spec.shape)
errors_spec = np.sqrt(errors_spec.T**2 + 4*circular_std(np.reshape(angles_NN_spec,[len(angles_NN_spec),-1]),axis=1)**2).T

muW_100, phiW_100, _, _ = t.fit_mod((angles_NN_spec[:,:,:10], angles_mom_spec, angles_sim_spec, moms_spec, errors_spec[:,:,:10]), 
                        method='weighted_MLE', error_weight=1.41)
print(muW_100, phiW_100)
mu_100, phi_100, _, _ = t.fit_mod((angles_NN_spec[:,:,:10], angles_mom_spec, angles_sim_spec, moms_spec, errors_spec[:,:,:10]))
print(mu_100, phi_100)

#     mus, phis = bootstrap((angles_NN_spec, angles_mom_spec, angles_sim_spec, moms_spec, errors_spec), args.B, error_weight=1.04)
#     mu1_total.append(mus)
#     phi1_total.append(phis)

#     angles_NN_spec = circular_mean(np.reshape(angles_NN_spec,[len(angles_NN_spec),-1]),axis=1)
#     errors_spec = np.sqrt(np.mean(np.reshape(errors_spec**2,[len(errors_spec),-1]),axis=1))

#     def mdp(lambd):
#         mu, _, Neff = weighted_stokes(angles_NN_spec, 1/errors_spec, lambd)
#         return MDP99(Neff,mu)

#     # res = minimize_scalar(mdp, bounds=(0,8),method="bounded")
#     mu, phi, Neff = weighted_stokes(angles_NN_spec, None, 0)
#     print("MDP >> ", mdp(0))
#     print("Mu >> ", mu)
#     print("Phi >> ", phi)
#     print("Neff >> ", Neff)
#     print("Lambda >> ", 0)
#     mdps.append(mdp(0))
#     mu1s.append(mu)
#     phi1s.append(phi)
#     Neff1s.append(Neff)
#     #lambda1s.append(res["x"])

# mu1_total = np.concatenate(mu1_total,axis=0)
# phi1_total = np.concatenate(phi1_total,axis=0)

# np.save("BOOT_POL2", (mu1_total, phi1_total))

# print("MDP final >> ",np.mean(mdps))
# print("Mu final >> ",np.mean(mu1s))
# print("Phi final >> ",np.mean(phi1s))
# print("Neff final >> ",np.mean(Neff1s))
# #print("Lambda final >> ",np.mean(lambda1s))




with open(home_dir + "review_unpol_train___flat_all_unpol__ensemble.pickle", "rb") as file:
    A = pickle.load(file)
angles, angles_mom, angles_sim, moms, errors, _, _, _, _, energies_sim, _, _, _ = A


mu0_total = []
phi0_total = []
for _ in range(args.pl_draws):
    Angles = (angles, angles_mom, angles_sim, moms, errors, energies_sim)
    angles_NN_spec, angles_mom_spec, angles_sim_spec, moms_spec, errors_spec, _ = generate_spectrum(Angles, f, fraction=0.85)
    errors_spec = np.sqrt(errors_spec.T**2 + 4*circular_std(np.reshape(angles_NN_spec,[len(angles_NN_spec),-1]),axis=1)**2).T
    print(angles_NN_spec.shape)

    mus, phis = bootstrap((angles_NN_spec[:,:,:10], angles_mom_spec, angles_sim_spec, moms_spec, errors_spec[:,:,:10]), args.B, error_weight=1.41)
    mu0_total.append(mus)
    phi0_total.append(phis)

mu0_total = np.concatenate(mu0_total,axis=0)
phi0_total = np.concatenate(phi0_total,axis=0)

np.save("BOOTUNPOL" + str(args.pl), (mu0_total, phi0_total))

# t = NetTest(n_nets=10)
# muW, phiW, _, _ = t.fit_mod((angles_NN_spec, angles_mom_spec, angles_sim_spec, moms_spec, errors_spec), 
#                             method='weighted_MLE', error_weight=1.04)
# muW, phiW, _ = weighted_stokes(np.ndarray.flatten(angles_NN_spec[:,:,:10]), 1/np.ndarray.flatten(errors_spec[:,:,:10]), 1.84)
# print("DONE: ", muW, phiW)
print("DONE")


# Neff0 = Neff_fit(mu0_total[:,0], phi0_total[:,0], (np.mean(Neff1s), muW, phiW), Nbound=2e5)
# Neff1 = Neff_fit(mu1_total[:,0], phi1_total[:,0], (np.mean(Neff1s), np.mean(mu1s), np.median(phi1s)), Nbound=2e5)
# print(f"Final bootstrap: Neff0 {Neff0[0]} -- Neff1 {Neff1[0]} -- mu1 {np.mean(mu1_total[:,0])} -- MDP0 {MDP99(Neff0[0],np.mean(mu1_total[:,0]))} -- MDP1 {MDP99(Neff1[0],np.mean(mu1_total[:,0]))} ")
# print(f"Final weighted stokes: Neff1 {np.mean(Neff1s)} -- mu1 {np.mean(mu1s)} -- MDP1 {np.mean(mdps)}")







###########################################################################
#For PLs

# Aeff= [49.8,62.61,68.92,81.09, 83.8,90.74,93.69,95.11, 
#        92.9,90.68,89.56,87.46,83.67,79.43, 74.2,72.45,69.46, 
#        65.7,62.69,59.65,56.27,52.82,49.53,46.54,44.29,42.03,39.61,
#        37.12,34.98,33.04,31.06,29.64,28.28,26.96,25.68,24.39,   
#        23,21.62,20.45,19.49,18.54,17.65,16.75,15.88,15.19,14.46,13.74,
#        13.27,12.81,12.12,11.07,10.12,9.243,8.403,7.758,7.132,6.521,5.937,
#        5.198,4.458,3.657,3.355,3.114,2.822,2.162,]

# Aeff = Aeff / np.max(Aeff)

# if args.pl == 1:
#     norm_factor = 1.552
# elif args.pl == 2:
#     norm_factor = 2.274
# else:
#     norm_factor = 1.0

# mu_samples = []
# muW025_samples = []
# muW05_samples = []
# muW075_samples = []
# muW1_samples = []
# muW125_samples = []
# muW15_samples = []
# muW175_samples = []
# muW2_samples = []

# chunks = []
# for _ in range(args.pl_draws):
#     # muW1_err_bootstrap = []
#     # muW_err_bootstrap = []
#     # mu_err_bootstrap = []
#     anglesE = []
#     angles_momE = []
#     momsE = []
#     errorsE = [] 
#     energies_simE = []
#     angles_simE = []

#     for i,e in enumerate(E[:]):
#         if e * (8.2 - 1.8) + 1.8 <= args.E[1] and e * (8.2 - 1.8) + 1.8 >= args.E[0]:
#             print(e * (8.2 - 1.8) + 1.8)
#             cut = (angles_sim[:,0] != 0) * (e == energies_sim[:,0])
#             if args.flat:
#                 pl_factor = int(np.sum(cut) * (e * (8.2 - 1.8) + 1.8)**(-args.pl) * 0.95)
#             else:
#                 pl_factor = int(np.sum(cut) * (e * (8.2 - 1.8) + 1.8)**(-args.pl) * norm_factor**2  * Aeff[i])
#             #pl_factor = int(np.sum(cut))
#             idx = np.random.choice(np.arange(np.sum(cut)), pl_factor, replace=False)
#             anglesE.append(angles[cut,:][idx,:])
#             angles_momE.append(angles_mom[:,0][cut][idx]) 
#             momsE.append(moms[:,0][cut][idx]) 
#             errorsE.append(errors[cut,:][idx,:] )
#             energies_simE.append(energies_sim[:,0][cut][idx])
#             angles_simE.append(angles_sim[:,0][cut][idx])
        
#     print("Total number of tracks >> ",np.concatenate(anglesE).ravel().shape[0] / 14)

#     n_cpu = os.cpu_count()
#     print("Beginning parallelization on {} cores\n".format(n_cpu))
#     chunks += [np.random.choice(np.arange(len(np.concatenate(angles_momE).ravel())),len(np.concatenate(angles_momE).ravel()), replace=True) for _ in range(args.B)]

#     def sub(idxs):
#         A1 = (np.concatenate(anglesE)[idxs].ravel(), np.concatenate(angles_momE).ravel()[idxs], np.concatenate(angles_simE).ravel()[idxs],
#         np.concatenate(momsE).ravel()[idxs], np.concatenate(errorsE)[idxs].ravel(), [None], [None], [None], [None], 
#         np.concatenate(energies_simE).ravel()[idxs],angles1,errors1,[None])
#         mu1, _, _, _ = t.fit_mod(A1, method='stokes')
#         mu2, _, _, _ = t.fit_mod(A1, method='weighted_MLE', error_weight=0.25)
#         mu3, _, _, _ = t.fit_mod(A1, method='weighted_MLE', error_weight=0.5)
#         mu4, _, _, _ = t.fit_mod(A1, method='weighted_MLE', error_weight=0.75)
#         mu5, _, _, _ = t.fit_mod(A1, method='weighted_MLE', error_weight=1)
#         mu6, _, _, _ = t.fit_mod(A1, method='weighted_MLE', error_weight=1.25)
#         mu7, _, _, _ = t.fit_mod(A1, method='weighted_MLE', error_weight=1.5)
#         mu8, _, _, _ = t.fit_mod(A1, method='weighted_MLE', error_weight=1.75)
#         mu9, _, _, _ = t.fit_mod(A1, method='weighted_MLE', error_weight=2)
#         return mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8, mu9 

#     with mp.Pool(processes=n_cpu) as pool:
#         results = pool.map(sub, chunks)
#     print("DONE!")
#     mu_err_bootstrap, muW025_err_bootstrap, muW05_err_bootstrap, muW075_err_bootstrap, muW1_err_bootstrap, muW125_err_bootstrap, muW15_err_bootstrap, muW175_err_bootstrap, muW2_err_bootstrap = zip(*results)
#     mu_samples += mu_err_bootstrap
#     muW025_samples += muW025_err_bootstrap
#     muW05_samples += muW05_err_bootstrap
#     muW075_samples += muW075_err_bootstrap
#     muW1_samples += muW1_err_bootstrap
#     muW125_samples += muW125_err_bootstrap
#     muW15_samples += muW15_err_bootstrap
#     muW175_samples += muW175_err_bootstrap
#     muW2_samples += muW2_err_bootstrap

# if args.pl_draws:
#     np.save("bootstrapSPL{}_draws{}_dist_flat{}_range{}_{}".format(args.pl, args.pl_draws, int(args.flat), args.E[0], args.E[1]), (mu_samples, muW025_samples, muW05_samples, muW075_samples, 
#                                                                                                                                     muW1_samples, muW125_samples, muW15_samples, muW175_samples, muW2_samples) )
# else:
#     np.save("bootstrapPL{}_dist".format(args.pl), (muW1_samples, muW_samples, mu_samples) )






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
