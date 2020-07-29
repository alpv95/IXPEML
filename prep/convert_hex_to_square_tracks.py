import numpy as np
import os
import torch
import copy as cp
import multiprocessing
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn.metrics.pairwise import pairwise_distances
from formats.dense_square import DenseSquareTracks, DenseSquareSimTracks
import multiprocess as mp
import h5py
import pdb

def rot60(x,y,idx):
    xy = np.stack([x,y])
    xy_rot = np.matmul(np.array([[np.cos(idx*np.pi/3),-np.sin(idx*np.pi/3)],[np.sin(idx*np.pi/3),np.cos(idx*np.pi/3)]]), xy)
    return xy_rot[0],xy_rot[1] 

def reflect(x,y,idx):
    if idx not in [0,1,2,3,4,5]: #identity
        return x,y 
    else:
        xy = np.stack([x,y])
        theta = 2 * (idx * np.pi/6)
        xy_mirror = np.matmul(np.array([[np.cos(theta),np.sin(theta)],[np.sin(theta),-np.cos(theta)]]), xy) 
        return xy_mirror[0], xy_mirror[1]

def hex2square_sub(hex_track, n_pixels, augment, shift):
    sim = False
    if hasattr(hex_track, 'angle'):
        sim = True

    if sim:
        angles_sq = np.zeros(augment)
        mom_phis_sq = np.zeros(augment)
        abs_pts_sq = np.zeros((augment, shift, 2))
        abs_pt_aug = np.empty(2)
    
    mom_abs_pt_aug = np.empty(2)
    mom_abs_pts_sq = np.zeros((augment, shift, 2)) 
    #tracks_cube = np.zeros((augment, shift, n_pixels, n_pixels), dtype=np.int16)  # tracks in square coords   

    xs = cp.copy(hex_track.x)
    ys = cp.copy(hex_track.y)
    Qs = cp.copy(hex_track.Q)

    ys = ys[np.isfinite(ys)] #- np.nanmin(hex_track.y)
    xs = xs[np.isfinite(xs)] #- np.nanmin(hex_track.x)
    Qs = Qs[Qs != -1]
    tracks_cube = np.zeros((augment, shift, 3, len(Qs)), dtype=np.int16)

    mom = cp.copy(hex_track.mom)
    mom_phi = cp.copy(hex_track.mom_phi)
    mom_abs_pt = cp.copy(hex_track.mom_abs_pt)
    #mom_abs_pt[0] -= np.nanmin(hex_track.x)
    #mom_abs_pt[1] -= np.nanmin(hex_track.y)

    if sim:
        abs_pt = cp.copy(hex_track.absorption_point)
        abs_pt = np.argmin(np.sqrt((xs - abs_pt[0])**2 +  (ys - abs_pt[1])**2))  
    mom_abs_pt = np.argmin(np.sqrt((xs - mom_abs_pt[0])**2 +  (ys - mom_abs_pt[1])**2)) 

    adj = np.argmin(np.sqrt((xs - xs[0])**2 +  (ys - ys[0])**2)[1:]) + 1 #find adjacent point
    r = np.sqrt((xs[0]-xs[adj])**2 + (ys[0]-ys[adj])**2) / 2
    a = r / np.cos(30*np.pi/180)

    if augment > 1:
        #n_rots = np.random.choice(np.arange(6), augment, replace=False)
        n_rots = np.array([0,2,4])
        n_flip = -1 #np.random.randint(0,12)
    else:
        n_rots = [0]
        n_flip = -1

    for k in range(augment):
        n_rot = n_rots[k]

        #reflect
        if sim:
            reflect_angle = reflect(np.cos(hex_track.angle),np.sin(hex_track.angle),n_flip)
            rotation = np.arctan2( reflect_angle[1], reflect_angle[0] ) - hex_track.angle
            rotation += n_rot * np.pi/3

        xs_aug, ys_aug = reflect(xs,ys,n_flip)
        
        #rotate
        xs_aug, ys_aug = rot60(xs_aug, ys_aug, n_rot)

        if sim:
            angle_new = hex_track.angle + rotation
            angle_new = np.mod(angle_new + np.pi ,2*np.pi) - np.pi
            mom_angle_new = hex_track.mom_phi + rotation
            mom_angle_new = np.mod(mom_angle_new + np.pi ,2*np.pi) - np.pi
            angles_sq[k] = angle_new
            mom_phis_sq[k] = mom_angle_new

        xs_aug_shift = np.where(np.mod(np.round((ys_aug - ys_aug.min())/(1.5*a)),2)==1, xs_aug+r, xs_aug) #(np.round(ys_aug,5) - 0.45465)/(1.5*a)),2)==1, xs_aug+r, xs_aug)
        j,i = np.round((ys_aug-ys_aug.min()) / (1.5*a)), np.round(((xs_aug_shift-xs_aug_shift.min())) / (2*r))
        if sim: 
            abs_pts_sq[k,0,0], abs_pts_sq[k, 0, 1] = i[abs_pt], j[abs_pt]
        mom_abs_pts_sq[k,0,0], mom_abs_pts_sq[k, 0, 1] = i[mom_abs_pt], j[mom_abs_pt]

        Q_square = np.zeros((n_pixels,n_pixels))
        bol = (i<n_pixels) * (j<n_pixels) #square crop to n_pixel size, a minority of high energy tracks will be sliced 
        Q_square[j[bol].astype(np.int),i[bol].astype(np.int)] = Qs[bol]
        indices = np.where(Q_square != 0)#sparse representation, hopefully doesnt slow down training
        values = Q_square[indices]
        tracks_cube[k, 0, :, :] = np.stack([*indices,values])#Q_square
        
        if shift > 1:
            xs_aug_shift = np.where(np.mod(np.round((ys_aug - ys_aug.min())/(1.5*a)),2)==0, xs_aug+r, xs_aug) #(np.round(ys_aug,5) - 0.45465)/(1.5*a)),2)==1, xs_aug+r, xs_aug)
            j,i = np.round((ys_aug-ys_aug.min()) / (1.5*a)), np.round(((xs_aug_shift-xs_aug_shift.min())) / (2*r))
            if sim: 
                abs_pts_sq[k,1,0], abs_pts_sq[k,1,1] = i[abs_pt], j[abs_pt]
            mom_abs_pts_sq[k,1,0], mom_abs_pts_sq[k,1,1] = i[mom_abs_pt], j[mom_abs_pt]
    
            Q_square = np.zeros((n_pixels,n_pixels))
            bol = (i<n_pixels) * (j<n_pixels) #square crop to n_pixel size, a minority of high energy tracks will be sliced 
            Q_square[j[bol].astype(np.int),i[bol].astype(np.int)] = Qs[bol]
            indices = np.where(Q_square != 0)#sparse representation, hopefully doesnt slow down training
            values = Q_square[indices]
            tracks_cube[k, 1, :, :] = np.stack([*indices,values]) #Q_square

    if sim:
        return tracks_cube, angles_sq, abs_pts_sq, mom_phis_sq, mom_abs_pts_sq
    else:
        return (tracks_cube, mom_phi, mom_abs_pts_sq)  


def hex2square(hex_tracks, n_pixels, augment=3, shift=2):# tracks_cum=None, angles_cum=None,
    if hasattr(hex_tracks, 'angles') and hasattr(hex_tracks, 'absorption_points'):
        sim = True
    else:
        sim = False

    n_cpu = os.cpu_count()
    print("Beginning parallelization on {} cores\n".format(n_cpu))
    chunks = [(hex_track, n_pixels, augment, shift) for hex_track in hex_tracks]
    #chunks = [hex_tracks[i::n_cpu] for i in range(n_cpu)] 
    with mp.Pool(processes=n_cpu) as pool:
        results = pool.starmap(hex2square_sub, chunks)
    print("DONE!")
    if sim: #Have to deal with Nones in list if skipping some tracks
        tracks_cum, angles_cum, abs_pts_cum, mom_phi_cum, mom_abs_pts_cum = zip(*results)
        angles_cum = torch.from_numpy(np.array(angles_cum).astype(np.float))
        abs_pts_cum = torch.from_numpy(np.array(abs_pts_cum).astype(np.float))
        mom_phi_cum = torch.from_numpy(np.array(mom_phi_cum).astype(np.float))
        mom_abs_pts_cum = torch.from_numpy(np.array(mom_abs_pts_cum).astype(np.float))
    else:
        tracks_cum, mom_phi_cum, mom_abs_pts_cum = zip(*results)
        mom_phi_cum = torch.from_numpy(np.array(mom_phi_cum).astype(np.float))
        mom_abs_pts_cum = torch.from_numpy(np.array(mom_abs_pts_cum).astype(np.float))

    print(mom_phi_cum.shape)
    print("Finished \n")

    if sim:
       return tracks_cum, angles_cum, mom_phi_cum, abs_pts_cum, mom_abs_pts_cum
    else:
       return tracks_cum, mom_phi_cum, mom_abs_pts_cum


# def sparse_hex_to_dense_square(hex_tracks, n_pixels, pitch, interp_method='linear', expand=True, n_cores=1):
#     """Convert to form ready for machine learning/deep neural nets and image processing

#     Angles and absorption points aren't changed by the conversion - but the dense tracks will need to be converted
#         back to coords using their centroids. So save the centroids for everything.

#     Note: The internals of the interpolation code also seem to be multithreaded (in an extension module) so it may not
#         be worth running with even more parallelism. However, this process is still slow.

#     Args:
#         hex_tracks: SparseHexTracks (or SparseHexSimTracks) tracks object
#         n_pixels: int, number of pixels width and height
#         pitch: double, spacing between pixel centers
#         interp_method: str, griddata's interpolation method. 'linear' seems to work well. Other methods like 'nearest'
#             may require more
#         expand: bool, whether to add dummy 0 points to allow proper interpolation with a compacted (0's removed)
#             sparse hex source. Default is True because the loading function that run before this set compact=True.
#         n_cores: int, number of cores to run in parallel. Default is 1.
#     """
#     # Split tracks among cores
#     n_tracks = hex_tracks.n_tracks
#     splits = np.array_split(np.arange(n_tracks), n_cores)

#     # Run in parallel
#     #with multiprocessing.Pool(processes=n_cores) as pool:
#         #results = pool.starmap(sparse_hex_to_dense_square_sub, [(hex_tracks[split], n_pixels, pitch, interp_method, expand) for split in splits])
#     #Run in a single process
#     results = sparse_hex_to_dense_square_sub(hex_tracks, n_pixels,pitch, interp_method, expand)

#     # Combine
#     #if hasattr(hex_tracks, 'angles') and hasattr(hex_tracks, 'absorption_points'):
#         #square_tracks = DenseSquareSimTracks.combine(results)
#     #else:
#         #square_tracks = DenseSquareTracks.combine(results)
#     square_tracks = results

#     return square_tracks


# def sparse_hex_to_dense_square_sub(hex_tracks, n_pixels, pitch, interp_method='linear', expand=True):
#     sim = False
#     if hasattr(hex_tracks, 'angles') and hasattr(hex_tracks, 'absorption_points'):
#         sim = True

#     n_tracks = hex_tracks.n_tracks
#     centroids = np.zeros((n_tracks, 2), dtype=np.float32)
#     n_pixels_half = int((n_pixels - 1) / 2)  # the number of pixels out from the center
#     tracks_cube = np.zeros((n_tracks, n_pixels, n_pixels), dtype=np.int16)  # tracks in square coords
#     for i_track, hex_track in enumerate(hex_tracks):
#         if i_track % 1000 == 0:
#             print('Interpolating track {}'.format(i_track))

#         xs = hex_track.x
#         ys = hex_track.y
#         Qs = hex_track.Q

#         # Dev/debug: Show starting hex
#         # plt.figure()
#         # plt.scatter(xs, ys, Qs)
#         # plt.axis('equal')
#         # plt.title('Hex {}'.format(i_track))
#         # plt.show()

#         # Do square grid conversion
#         # Calculate centroid
#         x0, y0 = sparse_weighted_centroid(xs, ys, Qs)

#         # Build square grid
#         x_square_ = np.linspace(x0 - n_pixels_half * pitch, x0 + n_pixels_half * pitch, n_pixels)
#         y_square_ = np.linspace(y0 - n_pixels_half * pitch, y0 + n_pixels_half * pitch, n_pixels)
#         x_square, y_square = np.meshgrid(x_square_, y_square_)

#         # Prep for interpolation by assigning 0 points outside the main "cluster"
#         #   Doing this here saves us from having to keep around all the grid 0's in the hex sparse datasets - but at the
#         #   cost of some imprecision here
#         if expand:
#             xy, Qs = add_zero_grid(xs, ys, Qs, x0, y0, n_pixels_half*pitch, n_pixels)  # n_pixels*2+1
#         else:
#             xy = np.column_stack((xs, ys))

#         # Do interpolation
#         Q_square = griddata(xy, Qs, (x_square, y_square), method=interp_method, fill_value=0).squeeze()

#         # Dev/debug: Show ending square interpolated
#         # plt.figure()
#         # plt.scatter(x_square.reshape(-1, 1), y_square.reshape(-1, 1), Q_square.reshape(-1,1))
#         # plt.axis('equal')
#         # plt.title('Square {}'.format(i_track))
#         # plt.show()

#         tracks_cube[i_track, :, :] = Q_square
#         centroids[i_track, :] = [x0, y0]

#     if sim:
#         square_tracks = DenseSquareSimTracks(tracks_cube, centroids, pitch, hex_tracks.angles, hex_tracks.absorption_points)
#     else:
#         square_tracks = DenseSquareTracks(tracks_cube, centroids, pitch)
#     return square_tracks


# def add_zero_grid(xs, ys, Qs, x0, y0, dist_half, n_pixels):
#     """Add a bunch of 0 points on a grid to the supplied (xs,ys) actual points. These are used for interpolation -
#     making sure points outside the cluster are 0.
#     Note/TODO: This is sort of an expensive way of doing this, especially the pairwise calcs
#     """
#     # Build fine grid - nominally matches the grid size that will be used for the interpolation
#     x_square_ = np.linspace(x0 - dist_half, x0 + dist_half, n_pixels)
#     y_square_ = np.linspace(y0 - dist_half, y0 + dist_half, n_pixels)
#     x_square, y_square = np.meshgrid(x_square_, y_square_)

#     # Make xy points
#     x = x_square.reshape(-1)
#     y = y_square.reshape(-1)
#     xy0 = np.column_stack((x, y))
#     xy = np.column_stack((xs, ys))

#     # Points close to any original point are close; others are far - keep the far points
#     D = pairwise_distances(xy, xy0)  # euclidean distance by default
#     d_cutoff = 0.07  # mm, same as dbscan neighbor cutoff
#     far = D > d_cutoff
#     keep = np.all(far, axis=0)
#     xy0 = xy0[keep, :]

#     # Assemble augmented cluster+0 points
#     xy = np.concatenate((xy, xy0), axis=0)
#     Q = np.concatenate((Qs, np.zeros((xy0.shape[0]))))
#     return xy, Q
