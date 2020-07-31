'''
Set of helpful methods for data manipulation.
'''
import numpy as np
from itertools import tee
import torch

def MDP99(N,mu100):
    ''' MDP99 for either Neff or N'''
    return 4.29 / (mu100*np.sqrt(N)) *100

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
    afactor = (eff1[:,1] * area[:,1]/0.85 * 3 * 
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
    afactor = (eff1[:,1] * area[:,1]/0.85 * 3 * 
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


def stokes(angles):
    N = len(angles)
    Q = sum(np.cos(2*angles)) / N
    U = sum(np.sin(2*angles)) / N
 
    return 2*np.sqrt(Q**2 + U**2), np.arctan2(U,Q)/2

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
