# Basic loss/objective function value calculations
#   This is just the basic components - the actual neural net modules are in the nn directory
import numpy as np


def wrap_angle(x):
    """Wrap angle into [-pi,pi). For numpy arrays. Broadcasts."""
    x = np.fmod(x + np.pi, 2 * np.pi)
    x = x + (x < 0) * 2 * np.pi
    return x - np.pi


def angular_distance(input, target):
    """Angle distance between input and target angles vectors' corresponding entries. For regular numpy arrays.
    Broadcasts. Results in radians in [-pi,pi)"""
    x = input - target

    x = np.fmod(x + np.pi, 2 * np.pi)
    x = x + (x < 0) * 2 * np.pi
    x = x - np.pi

    return x
