'''
Set of helpful methods for data manipulation.
'''
import os
import numpy as np
import multiprocess as mp
from itertools import tee
import pandas as pd
import torch
from scipy.optimize import minimize
from scipydirect import minimize as minimize_direct
from scipy.stats import norm
from util.net_test import *
from scipy.special import i0,i1
import scipy.integrate as integrate
import math

PIXEL_X_SPACE = 0.05 #in micrometers
PIXEL_Y_SPACE = 0.0433

######### General convenience functions ##########

def pi_pi(x):
    '''Bring angle (radians) to range [-pi,pi]'''
    return np.mod(x + np.pi,2*np.pi) - np.pi

def pi2_pi2(x):
    '''Bring angle (radians) to range [-pi/2, pi/2] '''
    return np.mod(x + np.pi/2,np.pi) - np.pi/2

def geo_mean(iterable, axis):
    '''Geometric mean'''
    a = np.log(iterable)
    return np.exp(a.mean(axis=axis))

def circular_mean(angles, weights, axis):
    '''Arg of first moment of Von Mises distribution'''
    mean = np.array([np.mean(weights*np.cos(2*angles), axis=axis), np.mean(weights*np.sin(2*angles), axis=axis)]) / np.sum(weights,axis=axis)
    return 0.5*np.arctan2(mean[1],mean[0])

def circular_std(angles, axis):
    ''' Circular variance can either be defined as var = 1-R [0,1], or var = -2ln(R) [0,inf]. 
        This function just returns R.
        For von Mises, concentration parameter k -> I1(k)/I0(k) = R.'''
    mean = np.array([np.mean(np.cos(2*angles),axis=axis), np.mean(np.sin(2*angles),axis=axis)])
    R = np.linalg.norm(mean,axis=0)
    return R

def mad(sample,side=0):
    '''Median absolute deviation'''
    med = np.median(sample)
    res = sample - med 
    if side > 0:
        out = np.median(res[res >= 0])
    elif side < 0:
        out = abs(np.median(res[res <= 0]))
    else:
        out = np.median(abs(res))
    return out

def fwhm(sample):
    '''full width half max'''
    n, bins, _ = plt.hist(sample,bins=160,density=True)
    plt.clf()
    mx = np.max(n)
    return np.sum((n >= mx/2) * abs(bins[1] - bins[2]))

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

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



############ Polarimetry Specific Convenience Functions ################



def MDP99(N,mu100):
    ''' MDP99 for either Neff or N'''
    return 4.29 / (mu100*np.sqrt(N)) *100

def stokes(angles):
    N = len(angles)
    Q = sum(np.cos(2*angles)) / N
    U = sum(np.sin(2*angles)) / N
 
    return 2*np.sqrt(Q**2 + U**2), np.arctan2(U,Q)/2

def weighted_qu(angles, weights=None, lambd=1, error=False):
    '''If weights equal (or lambd == 0) this reduces to normal Stokes, and Neff == N
    Normal Stokes estimation of Q and U are equivalent to the MLE.
    '''
    if weights is None:
        weights = np.ones(len(angles))
    Q = np.sum(2*weights**lambd*np.cos(2*angles))
    U = np.sum(2*weights**lambd*np.sin(2*angles))
    I = np.sum(weights**lambd)
    Neff = I**2 / np.sum(weights**(2*lambd))

    if error:
        #return covariance matrix
        V = (1/Neff) * np.array([[2 - (Q/I)**2, -Q*U/I**2],[-Q*U/I**2, 2 - (U/I)**2]])
        return Q/I, U/I, V
    return Q/I, U/I

def weighted_stokes(angles, weights=None, lambd=1, error=False):
    '''If weights equal (or lambd == 0) this reduces to normal Stokes, and Neff == N
    Normal Stokes estimation of Q and U are equivalent to the MLE.
    '''
    if weights is None:
        weights = np.ones(len(angles))
    Q = np.sum(2*weights**lambd*np.cos(2*angles))
    U = np.sum(2*weights**lambd*np.sin(2*angles))
    I = np.sum(weights**lambd)
    
    mu = np.sqrt(Q**2 + U**2) / I
    phi0 = 0.5*np.arctan2(U,Q)
    Neff = I**2 / np.sum(weights**(2*lambd))

    if error:
        #These are an approximation, use pol_posterior for proper errors if pi or mu very low
        mu_err = mu_error_gauss(mu, Neff)
        phi_err = phi_error_gauss(mu, Neff)
        return mu, phi0, Neff, mu_err, phi_err

    return mu, phi0, Neff

def pol_posterior(pi, phi, Neff, pi0=0, phi0=0, mu100=1):
    '''Full 2 dimensional posterior distribution for pi and phi, assuming uniform prior in Pi-phi space.'''
    sigma = np.sqrt(1/Neff * (1 - pi0**2*mu100**2/2))
    return np.sqrt(Neff) * pi*mu100**2 / (2*np.pi*sigma) * np.exp(-mu100**2/(4*sigma**2)
            *(pi0**2 + pi**2 -2*pi*pi0*np.cos(2*(phi - phi0)) - pi**2*pi0**2*mu100**2/2 
            * np.sin(2*(phi-phi0))**2))

def CL(Z, dx, dy=1, p=0.68, tol=0.005):
    '''
    Gets central Confidence level for contour plot for 2d or 1d posterior.
    Z_norm is the 2d or 1d grid of normalized posterior values. For 1d posterior keep dy=1.
    Returns posterior contour level.
    '''
    integral = np.sum(Z*dx*dy)
    if not math.isclose(integral,1):
        print("input posterior not normalized")
        Z = Z / integral
    x = 0
    max_iter = 200
    i = 0
    const = np.max(Z)/max_iter
    while np.abs(np.sum(Z[Z >= np.max(Z)-x]*dx*dy) - p) > tol and i < max_iter:
        x += const
        i += 1
    if i >= max_iter:
        print('Max iterations reached, need to lower tolerance.')
        return 
    return np.max(Z) - x

def upper_CL(Z, x, p=0.68, tol=0.005):
    '''For 1d distribution with minimum at zero
        Assumes sorted x and Z starting from 0.
        Returns actual upper limit.'''
    dx = x[1] - x[0]
    integral = np.sum(Z*dx)
    if not math.isclose(integral,1):
        print("input posterior not normalized")
        Z = Z / integral
    i = 0
    const = np.max(Z)/200
    while np.abs(np.sum(Z[x <= x[i]]*dx) - p) > tol and i < len(x):
        i += 1
    if i >= len(x):
        print('Max iterations reached, need to lower tolerance.')
        return 
    return x[i]

def mu_error_gauss(mu, Neff):
    '''Gaussian error on mu (modulation, mu=pi*mu_100) measurement, 
    gaussian approximation not appropriate when pi or mu low'''
    return np.sqrt((2 - mu**2)/(Neff-1))

def phi_error_gauss(mu, Neff):
    '''Gaussian error on phi measurement, usually fine unless mu or pi very low'''
    return 1 / (mu * np.sqrt(2*(Neff-1)))


def ellipticity_cut(angles, moms, keep_fraction):
        '''
        Applies ellipticity based cuts to predicted photoelectron angles.
        '''
        mom_cuts = np.arange(1,4,0.01)
        for mom_cut in mom_cuts:
            if len(angles[moms > mom_cut]) < keep_fraction * len(angles):
                break
        return mom_cut

def minimiseMDP(angles, weights):
    def mdp(lambd):
        mu, _, Neff = weighted_stokes(np.ndarray.flatten(angles), weights, lambd)
        return MDP99(Neff,mu)
    result = minimize_scalar(mdp, bounds=(0,8), method="bounded")
    return result['fun'], result['x']

def Aeff(E):
    '''
    Effective area as a function of energy at 687mbar
        Arguments: E (kev)
    '''
    try:
        area = np.loadtxt("notebooks/MMA_cal-eff-area_20200831.txt")[:,(0,-1)]
        eff1 = np.loadtxt("notebooks/du_efficiency_687mbar.txt")
    except OSError:
        area = np.loadtxt("MMA_cal-eff-area_20200831.txt")[:,(0,-1)]
        eff1 = np.loadtxt("du_efficiency_687mbar.txt")

    return np.interp(E, area[:,0],area[:,1]) * np.interp(E, eff1[:,0],eff1[:,1])

def exposure(spec, d, F):
    """
    Calculates total number of photons expected when observing a source
    with count spectrum spec for d days, with integrated flux density F erg/cm^2/s 
    between 1-9 keV.
    """
    
    spec_int, _ = integrate.quad(lambda e: spec(e)*e, 1, 9, limit=300)
    A = (F * 6.24151e+8) / spec_int
    N_obs = np.sum((d*24*60*60) * Aeff(np.linspace(1,9,801))
                   * A * spec(np.linspace(1,9,801)) 
                   * 0.01 )
    return N_obs

def paper_spec(E):
    '''Spectrum of energies for paper plots
    '''
    try:
        df_peri = pd.read_csv('notebooks/data/IXPE_tstspec/GX301_peri.txt',delim_whitespace=True)
        df_quad = pd.read_csv('notebooks/data/IXPE_tstspec/GX301_quad.txt',delim_whitespace=True)
        df_isp = pd.read_csv('notebooks/data/IXPE_tstspec/ISP.txt',delim_whitespace=True)
    except OSError:
        df_peri = pd.read_csv('data/IXPE_tstspec/GX301_peri.txt',delim_whitespace=True)
        df_quad = pd.read_csv('data/IXPE_tstspec/GX301_quad.txt',delim_whitespace=True)
        df_isp = pd.read_csv('data/IXPE_tstspec/ISP.txt',delim_whitespace=True)

    return np.maximum.reduce([np.interp(E, df_peri['ee'], df_peri['z']), np.interp(E, df_quad['ee'], 
                df_quad['z']), np.interp(E, df_isp['ee'], df_isp['z'])]) * Aeff(E)

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


def weightVM(sigma):
    w = i1(1/sigma**2) / i0(1/sigma**2)
    if isinstance(w,np.ndarray):
        return np.where(np.isnan(w),1,w)
    else:
        if np.isnan(w):
            return 1
        else:
            return w

def weightGauss(sigma):
    return np.exp(-2*sigma**2)

def weightPL(sigma):
    return 1/sigma

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


def Neff_fit(mus, phis, mle, Nbound=2e6, plot=False, pol=False):
    '''
    mle -> (N, mu, phi) from bootstrap distribution of mus and phis
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
        res.append(minimize(loglike_ ,x0,method="Nelder-Mead"))
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


############### Track Pre-Processing ##################


def sparse_mean(track_list, n_pixels):
    '''Mean and std of a set of sparse tracks, for training standardization.'''
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
    ''' Takes python list of sparse square tracks, converts them to dense arrays'''
    dense_tracks = []
    for track in sparse_tracks:
        dense_tracks.append(torch.sparse.FloatTensor(track[0,0,:2,:].long(), track[0,0,2,:], torch.Size([n_pixels,n_pixels])).to_dense())

    return torch.stack(dense_tracks)



##################### Tracks Postprocessing ######################



def triple_angle_reshape(inp, N, augment=3):
    '''Reshapes 1d ensemble list of results'''
    try:
        inp2 = np.reshape(inp,[-1,N],"F")
        inp3 = np.reshape(inp2,[-1,augment,N],"C")
    except:
        return inp
    return inp3

def triple_angle_rotate(ang):
    ''' Rotates augmented tracks back to their original orientation'''
    ang[:,1,:] -= 2*np.pi/3
    ang[:,2,:] += 2*np.pi/3
    return pi_pi(ang)

def square2hex_abs(abs_pts_sq, mom_abs_pts_sq, xy_abs_mom, num_pixels=50):
    '''Converts abs points from local image coordinates to global grid coords'''
    return num_pixels*(abs_pts_sq - mom_abs_pts_sq) * np.array([PIXEL_X_SPACE,PIXEL_Y_SPACE]) + xy_abs_mom

def optimal_weight(w, i=0):
    ''' Weight to mu100 and optimal weight conversion. 
    Recursive function that brings mu_100 to a linear relationship with weight.'''
    ps = [np.array([ 1.21410576e+01, -4.96954162e+01,  8.14789225e+01, -6.83102172e+01,
         3.11257214e+01, -7.62197920e+00,  1.88190338e+00,  7.76891394e-06]),
         np.array([ 7.51281699e+00, -2.67999731e+01,  3.77747179e+01, -2.65971436e+01,
                 9.68972542e+00, -1.68433976e+00,  1.10419005e+00,  6.09786434e-06]),
         np.array([ 3.98795853e-01, -1.45390707e+00,  2.10826041e+00, -1.54173524e+00,
                 5.91740029e-01, -1.10802105e-01,  1.00764230e+00,  5.90689355e-06]),
         np.array([ 3.55045046e-02, -1.27783735e-01,  1.82501225e-01, -1.30955776e-01,
                 4.89526689e-02, -8.75929102e-03,  1.00053458e+00,  5.89051327e-06]),
         np.array([-5.86490116e-04,  2.57522322e-03, -4.63776415e-03,  4.40756118e-03,
                -2.36393780e-03,  7.05504066e-04,  9.99894087e-01,  5.88811107e-06]),
         np.array([-3.30689955e-03,  1.25638640e-02, -1.92315072e-02,  1.51636514e-02,
                -6.52385298e-03,  1.48909836e-03,  9.99839831e-01,  5.88693882e-06]),
         np.array([-3.69052703e-03,  1.39149940e-02, -2.11173500e-02,  1.64838296e-02,
                -7.00476413e-03,  1.57361045e-03,  9.99834393e-01,  5.88587542e-06]),
         np.array([-3.65475003e-03,  1.37904198e-02, -2.09500759e-02,  1.63795325e-02,
                -6.97703089e-03,  1.57186220e-03,  9.99834229e-01,  5.88481068e-06])]
    if i == len(ps):
        return w
    return optimal_weight(np.dot(ps[i], np.array([w**7,w**6,w**5,w**4,w**3,w**2,w,1])),i+1)

def mu100_momE(Emom):
    ''' mom_E to mu100 conversion for unweighted moment analysis'''
    pE = np.array([-7.54382639e-06,  3.52854245e-04, -6.68021596e-03,  6.54785511e-02,
       -3.52446019e-01,  1.00884597e+00, -1.25207683e+00,  5.66052833e-01])
    return np.dot(pE,np.array([Emom**7,Emom**6,Emom**5,Emom**4,Emom**3,Emom**2,Emom,1]))

def error_combine(ang, sigma):
    '''Returns combined statistical and systematic uncertainty weights'''
    weight_epistemic = circular_std(ang, axis=(1,2))
    return optimal_weight(geo_mean(weightVM(sigma),axis=(1,2)) * weight_epistemic)

def pi_ambiguity_mean(ang, weight):
    '''Mean track angle from ensemble [-pi,pi]'''
    pi_fix = (np.mean((ang >= np.pi/2) + (ang < -np.pi/2), axis=(1,2)) >= 0.5) * np.pi
    return pi_pi(circular_mean(ang, weight, axis=(1,2)) + pi_fix)

def post_rotate(results_tuple, N, aug=3, datatype="sim", losstype='mserr1'):
    '''
    Takes output from gpu_test ensemble and re-rotates 3-fold angles appropriately. Also removes repeated outputs for moments.
    '''
    angles, angles_mom, angles_sim, moms, errors, abs_pts, mom_abs_pts, abs_pts_sim, \
    energies, energies_sim, energies_mom, zs, trgs, flags, p_tail, xy_abs_pts = results_tuple

    #reshape everybody
    ang = triple_angle_reshape(angles, N, augment=aug)
    mom = triple_angle_reshape(moms,N, augment=aug)
    E = triple_angle_reshape(energies_sim,N, augment=aug)
    E_nn = triple_angle_reshape(energies,N, augment=aug)
    E_mom = triple_angle_reshape(energies_mom,N, augment=1)
    error = triple_angle_reshape(errors, N, augment=aug)
    ang_mom = triple_angle_reshape(angles_mom,N, augment=aug)
    ang_sim = triple_angle_reshape(angles_sim,N, augment=aug)
    zs = triple_angle_reshape(zs, N, augment=1)
    p_tail = triple_angle_reshape(p_tail, N, augment=aug)
    trgs = triple_angle_reshape(trgs, N, augment=1)
    flags = triple_angle_reshape(flags, N, augment=1)

    #abs_pts get their own reshaping 
    xy_abs_pts = np.reshape(xy_abs_pts, [N,-1,2], "C")[0,:,:] 
    mom_abs_pts = np.mean(np.reshape(mom_abs_pts,[N,-1,aug,2],"C"),axis=0)[:,0,:] 
    
    if losstype == 'tailvpeak':
        p_tail = np.mean(p_tail, axis=(1,2))
        error_epis = [None]        
    else:
        E_nn = np.mean(E_nn, axis=(1,2)) 
        abs_pts = np.mean(np.reshape(abs_pts,[N,-1,aug,2],"C"),axis=0)[:,0,:]
        if aug == 3:
            ang = triple_angle_rotate(ang)
        #combine epistemic and aleatoric errors and average angles
        weight = error_combine(ang, error)
        ang = pi_ambiguity_mean(ang, 1/error**2)

    if datatype == "sim":
        abs_pts_sim = np.mean(np.reshape(abs_pts_sim,[N,-1,aug,2],"C"),axis=0)[:,0,:]

    if datatype == "meas":
        A = (ang, ang_mom[:,0,0], ang_sim, mom[:,0,0], weight, abs_pts, mom_abs_pts, abs_pts_sim, E_nn, E, E_mom[:,0,0], 
             zs, trgs[:,0,0], flags[:,0,0], p_tail, xy_abs_pts)
    else:
        A = (ang, ang_mom[:,0,0], ang_sim[:,0,0], mom[:,0,0], weight, abs_pts, mom_abs_pts, abs_pts_sim, E_nn, E[:,0,0], E_mom[:,0,0], 
            zs[:,0,0], trgs[:,0,0], flags[:,0,0], p_tail, xy_abs_pts)
    return A

def fits_save(results, file, datatype, losstype='mserr1'):
    '''Organizes final fits file save'''
    angles, angles_mom, angles_sim, moms, weights, abs_pts, mom_abs_pts, abs_pts_sim, \
    energies, energies_sim, energies_mom, zs, trgs, flags, p_tail, xy_abs_pts = results

    hdu = fits.PrimaryHDU()
    hdul = fits.HDUList([hdu])

    c2 = fits.Column(name='MOM_PHI', array=angles_mom, format='E',)
    c4 = fits.Column(name='MOM_ELLIP', array=moms, format='E',)
    c7 = fits.Column(name='MOM_ABS', array=mom_abs_pts, format='2E', dim='(2)')
    c8 = fits.Column(name='XY_MOM_ABS', array=xy_abs_pts, format='2E', dim='(2)')
    c14 = fits.Column(name='MOM_ENERGY', array=energies_mom, format='E')

    if losstype == 'tailvpeak':
        c17 = fits.Column(name='P_TAIL', array=p_tail, format='E')
        c3 = fits.Column(name='PHI', array=angles_sim, format='E',)
        c9 = fits.Column(name='ABS', array=abs_pts_sim, format='2E', dim='(3)')
        c13 = fits.Column(name='XYZ_ABS', array=np.concatenate((square2hex_abs(abs_pts_sim, mom_abs_pts, xy_abs_pts), np.expand_dims(zs,axis=-1)), axis=1), format='3E', dim='(3)')
        c10 = fits.Column(name='ENERGY', array=energies_sim, format='E')
        table_hdu = fits.BinTableHDU.from_columns([c17, c2, c4, c7, c8, c14, c3, c9, c13, c10])
        
    else:
        c1 = fits.Column(name='NN_PHI', array=angles, format='E')      
        c5 = fits.Column(name='NN_WEIGHT', array=weights, format='E')
        c6 = fits.Column(name='NN_ABS', array=abs_pts, format='2E', dim='(2)')
        c1 = fits.Column(name='NN_PHI', array=angles, format='E')
        c11 = fits.Column(name='NN_ENERGY', array=energies, format='E')
        c12 = fits.Column(name='XY_NN_ABS', array=square2hex_abs(abs_pts, mom_abs_pts, xy_abs_pts), format='2E', dim='(2)')
        if datatype == 'sim':
            c3 = fits.Column(name='PHI', array=angles_sim, format='E',)
            c9 = fits.Column(name='ABS', array=abs_pts_sim, format='2E', dim='(3)')
            c13 = fits.Column(name='XYZ_ABS', array=np.concatenate((square2hex_abs(abs_pts_sim, mom_abs_pts, xy_abs_pts), np.expand_dims(zs,axis=-1)), axis=1), format='3E', dim='(3)')
            c10 = fits.Column(name='ENERGY', array=energies_sim, format='E')
            table_hdu = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14])
        else:
            c15 = fits.Column(name='TRG_ID', array=trgs, format='J',)
            c18 = fits.Column(name='FLAG', array=flags, format='J',)
            table_hdu = fits.BinTableHDU.from_columns([c1, c2, c4, c5, c6, c7, c8, c11, c12, c14, c15, c18])

    hdul.append(table_hdu)
    hdul.writeto(file + '.fits', overwrite=True)
