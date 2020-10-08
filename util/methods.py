'''
Set of helpful methods for data manipulation.
'''
import os
import numpy as np
import multiprocess as mp
from itertools import tee
import torch
from ipopt import minimize_ipopt
from scipy.optimize import minimize
from scipydirect import minimize as minimize_direct
from scipy.stats import norm
from util.net_test import *

PIXEL_X_SPACE = 0.05 #in micrometers
PIXEL_Y_SPACE = 0.0433

def MDP99(N,mu100):
    ''' MDP99 for either Neff or N'''
    return 4.29 / (mu100*np.sqrt(N)) *100

def weighted_stokes(angles, weights, lambd):
    '''If weights equal (or lambd == 0) this reduces to normal Stokes, and Neff == N'''
    if weights is None:
        weights = np.ones(len(angles))
    Q = np.sum(2*weights**lambd*np.cos(2*angles))
    U = np.sum(2*weights**lambd*np.sin(2*angles))
    I = np.sum(weights**lambd)
    
    mu = np.sqrt(Q**2 + U**2) / I
    phi0 = 0.5*np.arctan2(U,Q)
    Neff = I**2 / np.sum(weights**(2*lambd))
    
    return mu, phi0, Neff

def ellipticity_cut(angles, moms, keep_fraction):
        '''
        Applies ellipticity based cuts to predicted photoelectron angles.
        '''
        mom_cuts = np.arange(1,4,0.01)
        for mom_cut in mom_cuts:
            if len(angles[moms > mom_cut]) < keep_fraction * len(angles):
                break
        return mom_cut

def pi_pi(x):
    '''Bring angle (radians) to range [-pi,pi]'''
    return np.mod(x + np.pi,2*np.pi) - np.pi

def pi2_pi2(x):
    '''Bring angle (radians) to range [-pi/2, pi/2] '''
    return np.mod(x + np.pi/2,np.pi) - np.pi/2

def Aeff(E):
    '''Effective area as a function of energy at 687mbar
        Arguments: E (kev)
    '''
    try:
        area = np.loadtxt("notebooks/Rev_IXPE_Mirror_Aeff.txt")
        eff1 = np.loadtxt("notebooks/du_efficiency_687mbar.txt")
    except OSError:
        area = np.loadtxt("Rev_IXPE_Mirror_Aeff.txt")
        eff1 = np.loadtxt("du_efficiency_687mbar.txt")
    afactor = (eff1[:,1] * area[:,1] * 3 * 
                0.04 * 10**(-10)/ (1.6022e-9))
    idx = np.argmin(abs(eff1[:,0] - E))
    return afactor[idx]

def Aeff_train(E):
    '''Effective area as a function of energy at 687mbar
        Arguments: E (kev)
    '''
    if E > 5.5: #flatten distribution edges for training
        E = 5.5
    elif E < 1.5:
        E = 1.5

    try:
        area = np.loadtxt("notebooks/Rev_IXPE_Mirror_Aeff.txt")
        eff1 = np.loadtxt("notebooks/du_efficiency_687mbar.txt")
    except OSError:
        area = np.loadtxt("Rev_IXPE_Mirror_Aeff.txt")
        eff1 = np.loadtxt("du_efficiency_687mbar.txt")
    afactor = (eff1[:,1] * area[:,1] * 3 * 
                0.04 * 10**(-10)/ (1.6022e-9))
    idx = np.argmin(abs(eff1[:,0] - E))
    return afactor[idx]

def triple_angle_reshape(inp, N, augment=3):
    try:
        inp2 = np.reshape(inp,[-1,N],"F")
        inp3 = np.reshape(inp2,[-1,augment,N],"C")
    except:
        return inp
    return inp3

def triple_angle_rotate(ang):
    ang[:,1,:] -= 2*np.pi/3
    ang[:,2,:] += 2*np.pi/3
    return pi_pi(ang)

def square2hex_abs(abs_pts_sq, mom_abs_pts_sq, xy_abs_mom):
    return (abs_pts_sq - mom_abs_pts_sq) * np.array([0.05,0.0433]) + xy_abs_mom

def circular_mean(angles, axis):
    mean = np.array([np.mean(np.cos(2*angles),axis=axis), np.mean(np.sin(2*angles),axis=axis)])
    mu = mean / np.linalg.norm(mean,axis=0)
    return 0.5*np.arctan2(mu[1],mu[0])

def circular_mean_weight(angles, weights, axis):
    mean = np.array([np.mean(weights*np.cos(2*angles),axis=axis), np.mean(weights*np.sin(2*angles),axis=axis)])
    mu = mean / np.linalg.norm(mean,axis=0)
    return 0.5*np.arctan2(mu[1],mu[0])

def circular_std(angles, axis):
    mean = np.array([np.mean(np.cos(2*angles),axis=axis), np.mean(np.sin(2*angles),axis=axis)])
    R = np.linalg.norm(mean,axis=0)
    return 0.5*np.sqrt((1 - R**2) / (R * (2 - R**2)))

def stokes(angles):
    N = len(angles)
    Q = sum(np.cos(2*angles)) / N
    U = sum(np.sin(2*angles)) / N
 
    return 2*np.sqrt(Q**2 + U**2), np.arctan2(U,Q)/2

def generate_spectrum(Angles, N_E, fraction=1):
    """
    Generates spectrum from set of NN, mom, sim angles (and associated errors,moments) given the spectral energy distribution N_E
    Fraction denotes fraction of tracks to use in making spectrum from the maximum possible
    """
    angles, angles_mom, angles_sim, moms, errors, energies_sim = Angles

    E = set(energies_sim[:])
    E = np.sort(list(E))
    max_tracks = len(angles_mom[energies_sim == E[10:71][np.argmax(N_E(E[10:71]))]]) / np.max(N_E(E[10:71]))
    print(max_tracks)

    angles_NN_spec = []
    angles_mom_spec = []
    angles_sim_spec = []
    moms_spec = []
    errors_spec = []
    energies_spec = []

    for i,e in enumerate(E[10:71]):
        cut = (e == energies_sim)
        N = int(N_E(e) * fraction * max_tracks)
        if i == 0 or e > 8.89:
            N = int(N*0.5)
        idxs = np.random.choice(np.arange(np.sum(cut)), size=N, replace=False)

        angles_NN_spec.append(angles[cut][idxs])
        angles_mom_spec.append(angles_mom[cut][idxs])
        angles_sim_spec.append(angles_sim[cut][idxs])
        errors_spec.append(errors[cut][idxs])
        moms_spec.append(moms[cut][idxs])
        energies_spec.append(energies_sim[cut][idxs])

    angles_NN_spec = np.concatenate(angles_NN_spec,axis=0)
    angles_mom_spec = np.concatenate(angles_mom_spec)
    angles_sim_spec = np.concatenate(angles_sim_spec)
    errors_spec = np.concatenate(errors_spec)
    moms_spec = np.concatenate(moms_spec)
    energies_spec = np.concatenate(energies_spec)

    return angles_NN_spec, angles_mom_spec, angles_sim_spec, moms_spec, errors_spec, energies_spec

def bootstrap(angles, B, error_weight=2, n_cpu=None):
    '''
    Non-parametric bootstrap
    angles -> distribution to be boostrapped (angles, angles_mom, angles_sim, moms, errors,)
    B -> number of boostrap samples
    '''
    t = NetTest(n_nets=10)
    if not n_cpu:
        n_cpu = os.cpu_count()
    print("Beginning parallelization on {} cores\n".format(n_cpu))
    chunks = []
    chunks += [np.random.choice(np.arange(len(angles[0])), len(angles[0]), replace=True) for _ in range(B)]
    def sub(idxs):
        A1 = (angles[0][idxs],angles[1][idxs],angles[2][idxs],angles[3][idxs],angles[4][idxs])
        mu, phi, _, _ = t.fit_mod(A1, method='stokes')
        muW, phiW, _, _ = t.fit_mod(A1, method='weighted_MLE', error_weight=error_weight)
        return mu, phi, muW, phiW
    with mp.Pool(processes=n_cpu) as pool:
        results = pool.map(sub, chunks)
    print("DONE!")
    mus, phis, muWs, phiWs = zip(*results)
    #mus, phis = zip(*results) #muWs, phiWs = zip(*results)
    MUs = np.concatenate([mus,muWs],axis=1)
    PHIs = np.concatenate([phis,phiWs],axis=1)
    # MUs = mus
    # PHIs = phis

    return MUs, PHIs

def BC(hist,mle):
    '''
    Bias-Correction for confidence intervals -- not really applicable to our case
    '''
    mu_hat = mle
    p0 = np.sum(hist < mu_hat) / len(hist)
    z0 = norm.ppf(p0)
    g_cdf = [np.sum(hist < hist[i]) / len(hist) if np.sum(hist < hist[i]) > 0 
             else np.sum(hist <= hist[i]) / len(hist) for i in range(len(hist))] 
    z_theta = np.array(norm.ppf(g_cdf)) - z0
    weights = norm.pdf(z_theta - z0) / norm.pdf(z_theta + z0)
    weights = weights / np.sum(weights)
    
    return weights

def Neff_fit(mus, phis, mle, Nbound=2e6, plot=False, pol=False):
    '''
    mle -> (N, mu, phi)
    '''
    loglike_ipopt = lambda x: -( len(mus)*(np.log(x[0]) - x[0]*x[1]**2 / 4)
                                         + np.sum( #np.log(mus) 
                                         - x[0]*mus**2 / 4  
                                         + x[0]*x[1]*mus * np.cos(2*(phis - x[2])) / 2 ) )
    bounds=[(0,Nbound),(0.,1),(-np.pi,np.pi)]
    res = []
    starts = [mle, (100000,0.01, 0),(1000,0.01,1),(40000,0.03,-1),(10000,0.001,0),(400000,0.001,1),(1000000,0.001,0),(1500000,0.001,1)]
    if pol:
        starts = [mle, (100000,0.5, 0),(1000,0.6,1),(40000,0.2,-1),(10000,0.3,0),(400000,1.0,1),(1000000,0.7,0),(1500000,0.8,1)]
        
    for x0 in starts:
        res.append(minimize(loglike_ipopt,x0,method="Nelder-Mead"))
        res.append(minimize_ipopt(loglike_ipopt, x0=x0, bounds=bounds, tol=1e-7))
    res.append(minimize_direct(loglike_ipopt, bounds=bounds,))
    Neff = [r["x"] for r in res if r["x"][1] >= 0.][np.argmin([r["fun"] for r in res if r["x"][1] >= 0.])] #return solution with minimum likelihood
    if plot:   
        dist = np.zeros(len(np.arange(0,0.02,0.0001)))
        for phi in np.linspace(-np.pi/2,np.pi/2,1000):
            dist += 2*np.pi/1000 * Neff[0]*np.arange(0,0.02,0.0001) / (4*np.pi*100) * np.exp(
            -Neff[0]*(np.arange(0,0.02,0.0001)**2 + Neff[1]**2 - 2*np.arange(0,0.02,0.0001)*Neff[1]*np.cos(2*(phi - Neff[2]))) / 4)
        return Neff, [np.arange(0,2,0.01), dist]
    else:
        return Neff

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def sparse_mean(track_list, n_pixels):
    dense_pad1 = np.zeros((n_pixels,n_pixels)) #one for each shift
    dense_pad2 = np.zeros((n_pixels,n_pixels))

    dense_pad1sq = np.zeros((n_pixels,n_pixels)) #for std
    dense_pad2sq = np.zeros((n_pixels,n_pixels))
    N = len(track_list)
    for track in track_list:
        indices1 = (track[0,0,0,:], track[0,0,1,:])
        values1 = track[0,0,2,:]
        dense_pad1[indices1] += values1.numpy()
        dense_pad1sq[indices1] += values1.numpy()**2

        indices2 = (track[0,1,0,:], track[0,1,1,:])
        values2 = track[0,1,2,:]
        dense_pad2[indices2] += values2.numpy()
        dense_pad2sq[indices2] += values2.numpy()**2

    mean = np.stack([dense_pad1 / N, dense_pad2 / N], axis=0)
    std = np.sqrt(np.stack([dense_pad1sq / N, dense_pad2sq / N], axis=0) - mean**2)
    return torch.from_numpy(mean), torch.from_numpy(std)

def sparse2dense(sparse_tracks, n_pixels=50):
    ''' Takes python list of sparse square tracks'''
    dense_tracks = []
    for track in sparse_tracks:
        dense_tracks.append(torch.sparse.FloatTensor(track[0,0,:2,:].long(), track[0,0,2,:], torch.Size([n_pixels,n_pixels])).to_dense())

    return torch.stack(dense_tracks)

def post_rotate(angles_tuple, N, aug=3, fix=False, datatype="sim"):
    '''
    Takes output from gpu_test ensemble and re-rotates 3-fold angles appropriately. Also removes repeated outputs for moments.
    '''
    angles, angles_mom, angles_sim, moms, errors, abs_pts, mom_abs_pts, abs_pts_sim, \
    energies, energies_sim, angles1, errors1, xy_abs_pts = angles_tuple

    ang = triple_angle_reshape(angles, N, augment=aug)
    mom = triple_angle_reshape(moms,N, augment=aug)
    E = triple_angle_reshape(energies_sim,N, augment=aug)
    E_nn = triple_angle_reshape(energies,N, augment=aug)
    error = triple_angle_reshape(errors,N, augment=aug)
    ang_mom = triple_angle_reshape(angles_mom,N, augment=aug)
    ang_sim = triple_angle_reshape(angles_sim,N, augment=aug)
    abs_pts = triple_angle_reshape(abs_pts,N, augment=aug)
    abs_pts_sim = triple_angle_reshape(abs_pts_sim,N, augment=aug)
    mom_abs_pts = triple_angle_reshape(mom_abs_pts,N, augment=aug)

    xy_abs_pts = np.reshape(xy_abs_pts, [-1,2,N], "C")[:,:,0]
    abs_pts = np.mean(np.reshape(abs_pts,[-1,2,aug,N],"C"),axis=-1)[:,:,0]
    mom_abs_pts = np.mean(np.reshape(mom_abs_pts,[-1,2,aug,N],"C"),axis=-1)[:,:,0]
    E_nn = np.mean(np.mean(E_nn[:,:,:],axis=2),axis=1) #E_nn[:,:,(1,2,4,5,6)],axis=2),axis=1)

    if datatype == "sim":
        abs_pts_sim = np.mean(np.reshape(abs_pts_sim,[-1,2,aug,N],"C"),axis=-1)[:,:,0]

    if aug == 3:
        ang = triple_angle_rotate(ang)

    # if datatype == "sim" and fix:
    #     ang[E > 6.4,:,:] = np.stack([ang[E > 6.4,0,:],ang[E > 6.4,0,:],ang[E > 6.4,0,:]],axis=1)

    #TODO: abs_pts and energy appropriate reshaping + condition if anything is [None]
    if datatype == "meas":
        A = (ang, ang_mom[:,0,0], ang_sim, mom[:,0,0], error, abs_pts, mom_abs_pts, abs_pts_sim, E_nn, E, angles1, errors1, xy_abs_pts)
    else:
        A = (ang, ang_mom[:,0,0], ang_sim[:,0,0], mom[:,0,0], error, abs_pts, mom_abs_pts, abs_pts_sim, E_nn, E[:,0,0], angles1, errors1, xy_abs_pts)
    return A
