import numpy as np
from util.methods import *
import sys
import os
sys.path.insert(0, '/home/groups/rwr/alpv95/tracksml')
from ipopt import minimize_ipopt
import pickle
import multiprocess as mp
import copy as cp
import argparse
from util.net_test import *

parser = argparse.ArgumentParser()
parser.add_argument('file', type=str,
                    help='Sample to bootstrap from')
parser.add_argument('B', type=int,
                    help='Number of boostrap samples')
args = parser.parse_args()

home_dir = '/home/groups/rwr/alpv95/tracksml/'
N = 14
NORM = 1.449 * 0.5 #0.1599
t = NetTest(n_nets=14)


with open(home_dir + args.file, "rb") as file:
    A = pickle.load(file)
angles, angles_mom, angles_sim, moms, errors, abs_pts, mom_abs_pts, abs_pts_sim, \
energies, energies_sim, _,_,_= A

ang = angles
mom = moms
E = energies_sim
error = errors
ang_mom = angles_mom
ang_sim = angles_sim
print(ang.shape)

def sub(idxs):
    A1 = (np.ndarray.flatten(ang[idxs]), ang_mom[idxs], ang_sim[idxs],
    mom[idxs], np.ndarray.flatten(error[idxs]), [None], [None], [None], [None], 
    E[idxs],[None],[None],[None])
    mu1, phi1, _, _ = t.fit_mod(A1, method='stokes')
    mu5, phi5, _, _ = t.fit_mod(A1, method='weighted_MLE', error_weight=1)
    mu9, phi9, _, _ = t.fit_mod(A1, method='weighted_MLE', error_weight=2)
    mu3, phi3, _, _ = t.fit_mod(A1, method='weighted_MLE', error_weight=1.5)
    return (mu1, phi1), (mu5, phi5), (mu9, phi9), (mu3, phi3)



pl_factor = []
idxs = []
for e in np.sort(list(set(E)))[5:]:
    idxss = list(set(np.where(E == e)[0]))
    idxs.append(idxss)
    n = len(idxss)
    pl_factor.append(int(n * Aeff(e) * NORM))

chunks = []
for b in range(args.B):
    jdxs = []
    for i,idx in enumerate(idxs):
        jdx = np.random.choice(idx, pl_factor[i], replace=True)
        jdxs.append(jdx)
    chunks += [np.concatenate(cp.copy(jdxs),axis=0)]


print("Total number of tracks >> ",np.concatenate(cp.copy(jdxs),axis=0).shape[0])

n_cpu = 18
print("Beginning parallelization on {} cores\n".format(n_cpu))

with mp.Pool(processes=n_cpu) as pool:
    results = pool.map(sub, chunks)
print("DONE!")
mu_err_bootstrap, muW1_err_bootstrap, muW2_err_bootstrap, muW3_err_bootstrap = zip(*results)

np.save("paperBoot_draws{}_range2_8".format(args.B),
         (mu_err_bootstrap, muW1_err_bootstrap, muW2_err_bootstrap, muW3_err_bootstrap) )
print("SAVED")