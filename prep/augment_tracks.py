# Functions for augmenting datasets and generating distributions of angles in an ensemble
import numpy as np
import matplotlib.pyplot as plt

def expand_tracks(tracks, n_final):
    """Expand (or contract) tracks to get a total size of n_final. If n_final <= n_tracks, random selection without
    replacement/selecting them all is done. If n_final > n_tracks, first all tracks are added; then remaining required
    slots are randomly sampled w/ replacement (hopefully this doesn't mess up the stats too much).

    Args:
        tracks: SparseHexTracks object (including SparseHexSimTracks) of input tracks
        n_final: int, number of expanded dataset tracks

    Returns:
        SparseHexTracks (or SparseHexSimTracks) object with n_final tracks
    """
    n_tracks = tracks.n_tracks
    if n_final <= n_tracks:
        inds = np.random.choice(n_tracks, size=n_final, replace=False)
        # For the case of n_final == n_tracks, this takes them all and shuffles them
    else:
        inds1 = np.arange(n_tracks)  # Ensure all tracks are sampled at least once
        inds2 = np.random.choice(n_tracks, size=(n_final-n_tracks), replace=True)
        inds = np.concatenate((inds1, inds2), axis=None)
        np.random.shuffle(inds)  # shuffle is in-place

    return tracks[inds]  # yes, this just works for SparseHexTracks

