'''
Set of helpful methods for data manipulation.
'''
import os
import numpy as np
import multiprocess as mp
from itertools import tee
import torch
from scipy.optimize import minimize
from scipy.stats import norm
from util.net_test import *
from scipy.special import i0, i1
import scipy.integrate as integrate
from scipy.optimize import minimize_scalar
import math

PIXEL_X_SPACE = 0.05  #in micrometers
PIXEL_Y_SPACE = 0.0433

######### General convenience functions ##########


def pi_pi(x):
    '''Bring angle (radians) to range [-pi,pi]'''
    return np.mod(x + np.pi, 2 * np.pi) - np.pi


def pi2_pi2(x):
    '''Bring angle (radians) to range [-pi/2, pi/2] '''
    return np.mod(x + np.pi / 2, np.pi) - np.pi / 2


def geo_mean(iterable, axis):
    '''Geometric mean'''
    a = np.log(iterable)
    return np.exp(a.mean(axis=axis))


def circular_mean(angles, weights, axis):
    '''Arg of first moment of Von Mises distribution'''
    mean = np.array([
        np.mean(weights * np.cos(2 * angles), axis=axis),
        np.mean(weights * np.sin(2 * angles), axis=axis)
    ])
    return 0.5 * np.arctan2(mean[1], mean[0])


def circular_std(angles, axis):
    ''' Circular variance can either be defined as var = 1-R [0,1], or var = -2ln(R) [0,inf]. 
        This function just returns R.
        For von Mises, concentration parameter k -> I1(k)/I0(k) = R.'''
    mean = np.array([
        np.mean(np.cos(2 * angles), axis=axis),
        np.mean(np.sin(2 * angles), axis=axis)
    ])
    R = np.linalg.norm(mean, axis=0)
    return R


def mad(sample, side=0):
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
    n, bins, _ = plt.hist(sample, bins=160, density=True)
    plt.clf()
    mx = np.max(n)
    return np.sum((n >= mx / 2) * abs(bins[1] - bins[2]))


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def BC(hist, mle):
    '''
    Bias-Correction for confidence intervals -- not really applicable to our case
    '''
    mu_hat = mle
    p0 = np.sum(hist < mu_hat) / len(hist)
    z0 = norm.ppf(p0)
    g_cdf = [
        np.sum(hist < hist[i]) /
        len(hist) if np.sum(hist < hist[i]) > 0 else np.sum(hist <= hist[i]) /
        len(hist) for i in range(len(hist))
    ]
    z_theta = np.array(norm.ppf(g_cdf)) - z0
    weights = norm.pdf(z_theta - z0) / norm.pdf(z_theta + z0)
    weights = weights / np.sum(weights)

    return weights


############ Polarimetry Specific Convenience Functions ################


def MDP99(N, mu100):
    ''' MDP99 for either Neff or N'''
    return 4.29 / (mu100 * np.sqrt(N)) * 100


def stokes(angles):
    N = len(angles)
    Q = sum(np.cos(2 * angles)) / N
    U = sum(np.sin(2 * angles)) / N

    return 2 * np.sqrt(Q**2 + U**2), np.arctan2(U, Q) / 2


def weighted_qu(angles, weights=None, lambd=1, error=False):
    '''If weights equal (or lambd == 0) this reduces to normal Stokes, and Neff == N
    Normal Stokes estimation of Q and U are equivalent to the MLE.
    '''
    if weights is None:
        weights = np.ones(len(angles))
    Q = np.sum(2 * weights**lambd * np.cos(2 * angles))
    U = np.sum(2 * weights**lambd * np.sin(2 * angles))
    I = np.sum(weights**lambd)
    Neff = I**2 / np.sum(weights**(2 * lambd))

    if error:
        #return covariance matrix
        V = (1 / Neff) * np.array([[2 - (Q / I)**2, -Q * U / I**2],
                                   [-Q * U / I**2, 2 - (U / I)**2]])
        return Q / I, U / I, V
    return Q / I, U / I


def weighted_stokes(angles, weights=None, lambd=1, error=False):
    '''If weights equal (or lambd == 0) this reduces to normal Stokes, and Neff == N
    Normal Stokes estimation of Q and U are equivalent to the MLE.
    '''
    if weights is None:
        weights = np.ones(len(angles))
    Q = np.sum(2 * weights**lambd * np.cos(2 * angles))
    U = np.sum(2 * weights**lambd * np.sin(2 * angles))
    I = np.sum(weights**lambd)

    mu = np.sqrt(Q**2 + U**2) / I
    phi0 = 0.5 * np.arctan2(U, Q)
    Neff = I**2 / np.sum(weights**(2 * lambd))

    if error:
        #These are an approximation, use pol_posterior for proper errors if pi or mu very low
        mu_err = mu_error_gauss(mu, Neff)
        phi_err = phi_error_gauss(mu, Neff)
        return mu, phi0, Neff, mu_err, phi_err

    return mu, phi0, Neff


def pol_posterior(pi, phi, Neff, pi0=0, phi0=0, mu100=1):
    '''Full 2 dimensional posterior distribution for pi and phi, assuming uniform prior in Pi-phi space.'''
    sigma = np.sqrt(1 / Neff * (1 - pi0**2 * mu100**2 / 2))
    return np.sqrt(Neff) * pi * mu100**2 / (2 * np.pi * sigma) * np.exp(
        -mu100**2 / (4 * sigma**2) *
        (pi0**2 + pi**2 - 2 * pi * pi0 * np.cos(2 * (phi - phi0)) -
         pi**2 * pi0**2 * mu100**2 / 2 * np.sin(2 * (phi - phi0))**2))


def CL(Z, dx, dy=1, p=0.68, tol=0.005):
    '''
    Gets Confidence level for contour plot for 2d posterior.
    Z_norm is the 2d grid of normalized posterior values
    '''
    integral = np.sum(Z * dx * dy)
    if not math.isclose(integral, 1):
        print("input posterior not normalized")
        Z = Z / integral
    x = 0
    max_iter = 1000
    i = 0
    const = np.max(Z) / max_iter
    while np.abs(np.sum(Z[Z >= np.max(Z) - x] * dx * dy) -
                 p) > tol and i < max_iter:
        x += const
        i += 1
    if i >= max_iter:
        print('Max iterations reached, need to lower tolerance.')
        return
    if dy == 1:  #1d posterior
        idxs = np.where(Z >= np.max(Z) - x)[0]
        return np.max(Z) - x, min(idxs), max(idxs)
    return np.max(Z) - x


def upper_CL(Z, x, p=0.68, tol=0.005):
    '''For 1d distribution with minimum at zero
        Assumes sorted x and Z starting from 0.
        Returns actual upper limit.'''
    dx = x[1] - x[0]
    integral = np.sum(Z * dx)
    if not math.isclose(integral, 1):
        print("input posterior not normalized")
        Z = Z / integral
    i = 0
    const = np.max(Z) / 200
    while np.abs(np.sum(Z[x <= x[i]] * dx) - p) > tol and i < len(x):
        i += 1
    if i >= len(x):
        print('Max iterations reached, need to lower tolerance.')
        return
    return x[i]


def mu_error_gauss(mu, Neff):
    '''Gaussian error on mu (modulation, mu=pi*mu_100) measurement, 
    gaussian approximation not appropriate when pi or mu low'''
    return np.sqrt((2 - mu**2) / (Neff - 1))


def phi_error_gauss(mu, Neff):
    '''Gaussian error on phi measurement, usually fine unless mu or pi very low'''
    return 1 / (mu * np.sqrt(2 * (Neff - 1)))


def ellipticity_cut(angles, moms, keep_fraction):
    '''
        Applies ellipticity based cuts to predicted photoelectron angles.
        '''
    mom_cuts = np.arange(1, 4, 0.01)
    for mom_cut in mom_cuts:
        if len(angles[moms > mom_cut]) < keep_fraction * len(angles):
            break
    return mom_cut


def MLError(self, angles, mu_hat, phi_hat):
    denom = (1 + mu_hat * np.cos(2 * (angles - phi_hat)))**2
    I00 = np.sum(np.cos(2 * (angles - phi_hat))**2 / denom)
    I01 = np.sum(2 * np.sin(2 * (angles - phi_hat)) / denom)
    I11 = np.sum(4 * mu_hat * (mu_hat + np.cos(2 * (angles - phi_hat))) /
                 denom)
    I = np.array([[I00, I01], [I01, I11]])
    I_1 = np.linalg.inv(I)
    return np.sqrt(I_1[0, 0]), np.sqrt(
        I_1[1, 1]), I_1[0, 1] / np.sqrt(I_1[0, 0] * I_1[1, 1])


def minimiseMDP(angles, weights):

    def mdp(lambd):
        mu, _, Neff = weighted_stokes(np.ndarray.flatten(angles), weights,
                                      lambd)
        return MDP99(Neff, mu)

    result = minimize_scalar(mdp, bounds=(0, 8), method="bounded")
    return result['fun'], result['x']


def Aeff(E):
    '''
    Effective area as a function of energy at 687mbar
        Arguments: E (kev)
    '''
    try:
        area = np.loadtxt("notebooks/MMA_cal-eff-area_20200831.txt")[:,
                                                                     (0, -1)]
        eff1 = np.loadtxt("notebooks/du_efficiency_687mbar.txt")
    except OSError:
        area = np.loadtxt("MMA_cal-eff-area_20200831.txt")[:, (0, -1)]
        eff1 = np.loadtxt("du_efficiency_687mbar.txt")

    return np.interp(E, area[:, 0], area[:, 1]) * np.interp(
        E, eff1[:, 0], eff1[:, 1])


def exposure(spec, d, F, INrange=(2, 8), OUTrange=(1, 10)):
    """
    Calculates total number of photons expected when observing a source
    with count spectrum spec for d days, with integrated flux density F erg/cm^2/s 
    between 1-9 keV.
    """
    assert isinstance(INrange, tuple)
    assert isinstance(OUTrange, tuple)

    spec_int, _ = integrate.quad(lambda e: spec(e) * e,
                                 INrange[0],
                                 INrange[1],
                                 limit=300)
    A = (F * 6.24151e+8) / spec_int
    x = np.linspace(OUTrange[0], OUTrange[1], 801)
    dx = x[1] - x[0]
    N_obs = np.sum((d * 24 * 60 * 60) * Aeff(x) * A * spec(x) * dx)
    return N_obs


def paper_spec(E):
    '''Spectrum of energies for paper plots
    '''
    try:
        df_peri = pd.read_csv('notebooks/data/IXPE_tstspec/GX301_peri.txt',
                              delim_whitespace=True)
        df_quad = pd.read_csv('notebooks/data/IXPE_tstspec/GX301_quad.txt',
                              delim_whitespace=True)
        df_isp = pd.read_csv('notebooks/data/IXPE_tstspec/ISP.txt',
                             delim_whitespace=True)
    except OSError:
        df_peri = pd.read_csv('data/IXPE_tstspec/GX301_peri.txt',
                              delim_whitespace=True)
        df_quad = pd.read_csv('data/IXPE_tstspec/GX301_quad.txt',
                              delim_whitespace=True)
        df_isp = pd.read_csv('data/IXPE_tstspec/ISP.txt',
                             delim_whitespace=True)

    return np.maximum.reduce([
        np.interp(E, df_peri['ee'], df_peri['z']),
        np.interp(E, df_quad['ee'], df_quad['z']),
        np.interp(E, df_isp['ee'], df_isp['z'])
    ]) * Aeff(E)


def Aeff_train(E):
    '''Effective area as a function of energy at 687mbar
        Arguments: E (kev)
    '''
    if E > 5.5:  #flatten distribution edges for training
        E = 5.5
    elif E < 1.5:
        E = 1.5

    try:
        area = np.loadtxt("notebooks/Rev_IXPE_Mirror_Aeff.txt")
        eff1 = np.loadtxt("notebooks/du_efficiency_687mbar.txt")
    except OSError:
        area = np.loadtxt("Rev_IXPE_Mirror_Aeff.txt")
        eff1 = np.loadtxt("du_efficiency_687mbar.txt")
    afactor = (eff1[:, 1] * area[:, 1] * 3 * 0.04 * 10**(-10) / (1.6022e-9))
    idx = np.argmin(abs(eff1[:, 0] - E))
    return afactor[idx]


def weightVM(sigma):
    w = i1(1 / sigma**2) / i0(1 / sigma**2)
    if isinstance(w, np.ndarray):
        return np.where(np.isnan(w), 1, w)
    else:
        if np.isnan(w):
            return 1
        else:
            return w


def weightGauss(sigma):
    return np.exp(-2 * sigma**2)


def weightPL(sigma):
    return 1 / sigma


def generate_spectrum(Angles, N_E, fraction=1):
    """
    Generates spectrum from set of NN, mom, sim angles (and associated errors,moments) given the spectral energy distribution N_E
    Fraction denotes fraction of tracks to use in making spectrum from the maximum possible
    """
    angles, angles_mom, angles_sim, moms, errors, energies_sim = Angles

    E = set(energies_sim[:])
    E = np.sort(list(E))
    max_tracks = len(angles_mom[energies_sim == E[10:71][np.argmax(
        N_E(E[10:71]))]]) / np.max(N_E(E[10:71]))
    print(max_tracks)

    angles_NN_spec = []
    angles_mom_spec = []
    angles_sim_spec = []
    moms_spec = []
    errors_spec = []
    energies_spec = []

    for i, e in enumerate(E[10:71]):
        cut = (e == energies_sim)
        N = int(N_E(e) * fraction * max_tracks)
        if i == 0 or e > 8.89:
            N = int(N * 0.5)
        idxs = np.random.choice(np.arange(np.sum(cut)), size=N, replace=False)

        angles_NN_spec.append(angles[cut][idxs])
        angles_mom_spec.append(angles_mom[cut][idxs])
        angles_sim_spec.append(angles_sim[cut][idxs])
        errors_spec.append(errors[cut][idxs])
        moms_spec.append(moms[cut][idxs])
        energies_spec.append(energies_sim[cut][idxs])

    angles_NN_spec = np.concatenate(angles_NN_spec, axis=0)
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
    chunks += [
        np.random.choice(np.arange(len(angles[0])),
                         len(angles[0]),
                         replace=True) for _ in range(B)
    ]

    def sub(idxs):
        A1 = (angles[0][idxs], angles[1][idxs], angles[2][idxs],
              angles[3][idxs], angles[4][idxs])
        mu, phi, _, _ = t.fit_mod(A1, method='stokes')
        muW, phiW, _, _ = t.fit_mod(A1,
                                    method='weighted_MLE',
                                    error_weight=error_weight)
        return mu, phi, muW, phiW

    with mp.Pool(processes=n_cpu) as pool:
        results = pool.map(sub, chunks)
    print("DONE!")
    mus, phis, muWs, phiWs = zip(*results)
    #mus, phis = zip(*results) #muWs, phiWs = zip(*results)
    MUs = np.concatenate([mus, muWs], axis=1)
    PHIs = np.concatenate([phis, phiWs], axis=1)
    # MUs = mus
    # PHIs = phis

    return MUs, PHIs


def Neff_fit(mus, phis, mle, Nbound=2e6, plot=False, pol=False):
    '''
    mle -> (N, mu, phi) from bootstrap distribution of mus and phis
    '''
    loglike_ipopt = lambda x: -(
        len(mus) * (np.log(x[0]) - x[0] * x[1]**2 / 4) + np.sum(  #np.log(mus) 
            -x[0] * mus**2 / 4 + x[0] * x[1] * mus * np.cos(2 * (phis - x[2]))
            / 2))
    bounds = [(0, Nbound), (0., 1), (-np.pi, np.pi)]
    res = []
    starts = [
        mle, (100000, 0.01, 0), (1000, 0.01, 1), (40000, 0.03, -1),
        (10000, 0.001, 0), (400000, 0.001, 1), (1000000, 0.001, 0),
        (1500000, 0.001, 1)
    ]
    if pol:
        starts = [
            mle, (100000, 0.5, 0), (1000, 0.6, 1), (40000, 0.2, -1),
            (10000, 0.3, 0), (400000, 1.0, 1), (1000000, 0.7, 0),
            (1500000, 0.8, 1)
        ]

    for x0 in starts:
        res.append(minimize(loglike_, x0, method="Nelder-Mead"))
        res.append(
            minimize_ipopt(loglike_ipopt, x0=x0, bounds=bounds, tol=1e-7))
    res.append(minimize_direct(
        loglike_ipopt,
        bounds=bounds,
    ))
    Neff = [r["x"] for r in res if r["x"][1] >= 0.
            ][np.argmin([r["fun"] for r in res if r["x"][1] >= 0.
                         ])]  #return solution with minimum likelihood
    if plot:
        dist = np.zeros(len(np.arange(0, 0.02, 0.0001)))
        for phi in np.linspace(-np.pi / 2, np.pi / 2, 1000):
            dist += 2 * np.pi / 1000 * Neff[0] * np.arange(
                0, 0.02, 0.0001) / (4 * np.pi * 100) * np.exp(
                    -Neff[0] * (np.arange(0, 0.02, 0.0001)**2 + Neff[1]**2 -
                                2 * np.arange(0, 0.02, 0.0001) * Neff[1] *
                                np.cos(2 * (phi - Neff[2]))) / 4)
        return Neff, [np.arange(0, 2, 0.01), dist]
    else:
        return Neff


############### Track Pre-Processing ##################


def sparse_mean(track_list, n_pixels):
    '''Mean and std of a set of sparse tracks, for training standardization.'''
    dense_pad1 = np.zeros((n_pixels, n_pixels))  #one for each shift
    dense_pad2 = np.zeros((n_pixels, n_pixels))

    dense_pad1sq = np.zeros((n_pixels, n_pixels))  #for std
    dense_pad2sq = np.zeros((n_pixels, n_pixels))
    N = len(track_list)
    for track in track_list:
        indices1 = (track[0, 0, 0, :], track[0, 0, 1, :])
        values1 = track[0, 0, 2, :]
        dense_pad1[indices1] += values1.numpy()
        dense_pad1sq[indices1] += values1.numpy()**2

        indices2 = (track[0, 1, 0, :], track[0, 1, 1, :])
        values2 = track[0, 1, 2, :]
        dense_pad2[indices2] += values2.numpy()
        dense_pad2sq[indices2] += values2.numpy()**2

    mean = np.stack([dense_pad1 / N, dense_pad2 / N], axis=0)
    std = np.sqrt(
        np.stack([dense_pad1sq / N, dense_pad2sq / N], axis=0) - mean**2)
    return torch.from_numpy(mean), torch.from_numpy(std)


def sparse2dense(sparse_tracks, n_pixels=50):
    ''' Takes python list of sparse square tracks, converts them to dense arrays'''
    dense_tracks = []
    for track in sparse_tracks:
        dense_tracks.append(
            torch.sparse.FloatTensor(track[0, 0, :2, :].long(), track[0, 0,
                                                                      2, :],
                                     torch.Size([n_pixels,
                                                 n_pixels])).to_dense())

    return torch.stack(dense_tracks)


##################### Tracks Postprocessing ######################


def triple_angle_reshape(inp, N, augment=3):
    '''Reshapes 1d ensemble list of results'''
    try:
        inp2 = np.reshape(inp, [-1, N], "F")
        inp3 = np.reshape(inp2, [-1, augment, N], "C")
    except:
        return inp
    return inp3


def triple_angle_rotate(ang):
    ''' Rotates augmented tracks back to their original orientation'''
    if ang.shape[1] == 3:
        ang[:, 1, :] -= 2 * np.pi / 3
        ang[:, 2, :] += 2 * np.pi / 3
    elif ang.shape[1] == 6:
        ang[:, 1, :] -= 1 * np.pi / 3
        ang[:, 2, :] -= 2 * np.pi / 3
        ang[:, 3, :] -= np.pi
        ang[:, 4, :] -= 4 * np.pi / 3
        ang[:, 5, :] -= 5 * np.pi / 3
    else:
        raise ('Error, wrong shape for angles.')
    return pi_pi(ang)


def square2hex_abs(abs_pts_sq, mom_abs_pts_sq, xy_abs_mom, num_pixels=50):
    '''Converts abs points from local image coordinates to global grid coords'''
    return num_pixels * (abs_pts_sq - mom_abs_pts_sq) * np.array(
        [PIXEL_X_SPACE, PIXEL_Y_SPACE]) + xy_abs_mom


def error_combine(ang, sigma):
    '''Returns combined statistical and systematic uncertainty weights'''
    weight_epistemic = circular_std(ang, axis=(1, 2))
    return geo_mean(weightVM(sigma),
                    axis=(1, 2)), weight_epistemic  #optimal weight


def error_combine_gauss(ang, sigma):
    '''Returns combined statistical and systematic uncertainty weights'''
    weight_epistemic = circular_std(ang, axis=(1, 2))
    return 1 / np.mean(sigma**2,
                       axis=(1, 2)), weight_epistemic  #optimal weight


def pi_ambiguity_mean(ang, weight, seed=None):
    '''Mean track angle from ensemble [-pi,pi]. Voting algorithm for principal axis direction.'''
    vote = np.mean((ang >= np.pi / 2) + (ang < -np.pi / 2), axis=(1, 2))
    if seed is not None:
        np.random.seed(seed)
    pi_fix = np.random.randint(2, size=vote.shape) * np.pi
    pi_fix[vote > 0.5] = np.pi
    pi_fix[vote < 0.5] = 0
    no_direction_flag = (vote == 0.5) * 1
    return pi_pi(circular_mean(ang, weight, axis=(1, 2)) +
                 pi_fix), no_direction_flag


def post_rotate(results_tuple,
                N,
                aug,
                datatype="sim",
                losstype='mserr1',
                seed=None):
    '''
    Takes output from gpu_test ensemble and re-rotates 3-fold angles appropriately. Also removes repeated outputs for moments.
    '''
    angles, angles_mom, angles_sim, moms, errors, abs_pts, mom_abs_pts, abs_pts_sim, \
    energies, energies_sim, energies_mom, zs, trgs, flags, p_tail, xy_abs_pts, xy_abs_pts_true = results_tuple

    #reshape everybody
    ang = triple_angle_reshape(angles, N, augment=aug)
    mom = triple_angle_reshape(moms, N, augment=aug)
    E = triple_angle_reshape(energies_sim, N, augment=aug)
    E_nn = triple_angle_reshape(energies, N, augment=aug)
    E_mom = triple_angle_reshape(energies_mom, N, augment=1)
    error = triple_angle_reshape(errors, N, augment=aug)
    ang_mom = triple_angle_reshape(angles_mom, N, augment=aug)
    ang_sim = triple_angle_reshape(angles_sim, N, augment=aug)
    zs = triple_angle_reshape(zs, N, augment=1)
    p_tail = triple_angle_reshape(p_tail, N, augment=aug)
    trgs = triple_angle_reshape(trgs, N, augment=1)
    flags = triple_angle_reshape(flags, N, augment=1)

    #abs_pts get their own reshaping
    xy_abs_pts = np.reshape(xy_abs_pts, [N, -1, 2], "C")[0, :, :]
    mom_abs_pts = np.mean(np.reshape(mom_abs_pts, [N, -1, aug, 2], "C"),
                          axis=0)[:, 0, :]

    if losstype == 'tailvpeak' or losstype == 'energy':
        p_tail = np.mean(p_tail, axis=(1, 2))
        weight = [None]
        weight_epis = [None]
    else:
        E_nn = np.mean(E_nn, axis=(1, 2))
        abs_pts = np.mean(np.reshape(abs_pts, [N, -1, aug, 2], "C"),
                          axis=0)[:, 0, :]
        if aug == 3 or aug == 6:
            ang = triple_angle_rotate(ang)
        #combine epistemic and aleatoric errors and average angles
        weight, weight_epis = error_combine(ang, error)
        ang, no_direction_flag = pi_ambiguity_mean(ang, weight=1, seed=seed)

    if datatype == "sim":
        abs_pts_sim = np.mean(np.reshape(abs_pts_sim, [N, -1, aug, 2], "C"),
                              axis=0)[:, 0, :]
        xy_abs_pts_true = np.reshape(xy_abs_pts_true, [N, -1, 2], "C")[0, :, :]

    if datatype == "meas":
        A = (ang, ang_mom[:, 0,
                          0], ang_sim, mom[:, 0,
                                           0], weight, weight_epis, abs_pts,
             mom_abs_pts, abs_pts_sim, E_nn, E, E_mom[:, 0, 0], zs, trgs[:, 0,
                                                                         0],
             flags[:, 0,
                   0], p_tail, xy_abs_pts, xy_abs_pts_true, no_direction_flag)
    else:
        A = (ang, ang_mom[:, 0, 0], ang_sim[:, 0, 0], mom[:, 0, 0], weight,
             weight_epis, abs_pts, mom_abs_pts, abs_pts_sim, E_nn, E[:, 0, 0],
             E_mom[:, 0, 0], zs[:, 0, 0], trgs[:, 0, 0], flags[:, 0, 0],
             p_tail, xy_abs_pts, xy_abs_pts_true, no_direction_flag)
    return A


def fits_save(results, p_tail, file, datatype, losstype='mserr1'):
    '''Organizes final fits file save'''
    angles, angles_mom, angles_sim, moms, weights, weights_epis, abs_pts, mom_abs_pts, abs_pts_sim, \
    energies, energies_sim, energies_mom, zs, trgs, flags, _, xy_abs_pts, xy_abs_pts_true, no_direction_flag = results

    hdu = fits.PrimaryHDU()
    hdul = fits.HDUList([hdu])

    c2 = fits.Column(
        name='MOM_PHI',
        array=angles_mom,
        format='E',
    )
    c4 = fits.Column(
        name='MOM_ELLIP',
        array=moms,
        format='E',
    )
    c7 = fits.Column(name='MOM_ABS', array=mom_abs_pts, format='2E', dim='(2)')
    c8 = fits.Column(name='XY_MOM_ABS',
                     array=xy_abs_pts,
                     format='2E',
                     dim='(2)')
    c14 = fits.Column(name='MOM_ENERGY', array=energies_mom, format='E')
    c1 = fits.Column(name='NN_PHI', array=angles, format='E')
    c5 = fits.Column(name='NN_WEIGHT', array=weights, format='E')
    cEpis = fits.Column(name='NN_WEIGHT_EPIS', array=weights_epis, format='E')
    c6 = fits.Column(name='NN_ABS', array=abs_pts, format='2E', dim='(2)')
    c1 = fits.Column(name='NN_PHI', array=angles, format='E')
    c11 = fits.Column(name='NN_ENERGY', array=energies, format='E')
    c12 = fits.Column(name='XY_NN_ABS',
                      array=square2hex_abs(abs_pts, mom_abs_pts, xy_abs_pts),
                      format='2E',
                      dim='(2)')
    c17 = fits.Column(name='P_TAIL', array=p_tail, format='E')
    c18 = fits.Column(
        name='FLAG',
        array=flags,
        format='J',
    )
    c19 = fits.Column(
        name='DIRECTION_FLAG',
        array=no_direction_flag,
        format='J',
    )
    if datatype == 'sim':
        c3 = fits.Column(
            name='PHI',
            array=angles_sim,
            format='E',
        )
        c9 = fits.Column(name='ABS', array=abs_pts_sim, format='2E', dim='(2)')
        c13 = fits.Column(name='XYZ_ABS',
                          array=np.concatenate(
                              (xy_abs_pts_true, np.expand_dims(zs, axis=-1)),
                              axis=1),
                          format='3E',
                          dim='(3)')
        c10 = fits.Column(name='ENERGY', array=energies_sim, format='E')
        table_hdu = fits.BinTableHDU.from_columns([
            c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, cEpis,
            c17, c18, c19
        ])
    else:
        c15 = fits.Column(
            name='TRG_ID',
            array=trgs,
            format='J',
        )
        table_hdu = fits.BinTableHDU.from_columns([
            c1, c2, c4, c5, c6, c7, c8, c11, c12, c14, c15, cEpis, c17, c18,
            c19
        ])

    hdul.append(table_hdu)
    hdul.writeto(file + '.fits', overwrite=True)
