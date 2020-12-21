# Randomize and train/validate/test split data for track angle prediction
import numpy as np
from sklearn.model_selection import train_test_split


def tvt_random_split(inds, fracs=(0.7, 0.15, 0.15), rng_seed=None):
    """Split data into 3 sets randomly

    Args:
        inds: array of indexes to split
        fracs: (train frac, validate frac, test frac) tuple of floats that add to 1
        rng_seed: Optional RNG seed for consistency

    Returns:
        (train inds, validate inds, test inds) arrays of split indexes
    """
    # Set RNG seed if present
    if rng_seed is not None:
        np.random.seed(rng_seed)

    # Use sklearn's split fun twice
    split1 = train_test_split(inds, test_size=fracs[2])  # returns [tv, t]
    split2 = train_test_split(split1[0], test_size=fracs[1] / (fracs[0] + fracs[1]))  # returns [t, v]
    return split2[0], split2[1], split1[1]


def tt_random_split(inds, fracs=(0.7, 0.3), rng_seed=None):
    """Split data into 2 sets randomly

    Args:
        inds: array of indexes to split
        fracs: (train frac, test frac) tuple of floats that add to 1
        rng_seed: Optional RNG seed for consistency

    Returns:
        (train inds, test inds) arrays of split indexes
    """
    # Set RNG seed if present
    if rng_seed is not None:
        np.random.seed(rng_seed)

    split = train_test_split(inds, test_size=fracs[1], shuffle=False)  # returns [t, t]
    return split[0], split[1]
