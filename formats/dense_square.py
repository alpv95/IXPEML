import h5py
import numpy as np


class DenseSquareTracks:
    """Struct of arrays form of multiple tracks stored as arrays of images.
    Performance note: it's fast to construct this from 3 lists of components, but would be slow to construct track-by-
    track. It's fast to get a single track from this, but would be slow to insert/delete tracks. Only the fast methods
    are implemented.
    """

    def __init__(self, tracks_cube, centroids, pitch, rotations):
        self.tracks_cube = tracks_cube
        self.centroids = centroids
        self.pitch = pitch
        self.rotations = rotations
        self.n_tracks = tracks_cube.shape[0]
        n_pixels = tracks_cube.shape[2]
        self.tracks_max = np.max(np.reshape(tracks_cube,[-1,n_pixels,n_pixels]).astype(np.float32), axis=0)
        self.tracks_mean = np.mean(np.divide(np.reshape(tracks_cube,[-1,n_pixels,n_pixels]).astype(np.float32), self.tracks_max, out=np.zeros_like(np.reshape(tracks_cube,[-1,n_pixels,n_pixels]).astype(np.float32)), where=self.tracks_max!=0), axis=0) #np.mean(tracks_cube.astype(np.float32) / self.tracks_max, axis=0)
        self.tracks_std = np.std(np.divide(np.reshape(tracks_cube,[-1,n_pixels,n_pixels]).astype(np.float32), self.tracks_max, out=np.zeros_like(np.reshape(tracks_cube,[-1,n_pixels,n_pixels]).astype(np.float32)), where=self.tracks_max!=0), axis=0) #np.std(tracks_cube.astype(np.float32) / self.tracks_max, axis=0)

    def save(self, filename=None, f=None):
        """Save to HDF5 file. Either specify a filename to save a single object or f to save multiple objects"""
        if f is None:
            with h5py.File(filename, 'w') as f:
                _save_tracks(self, f)
        else:
            _save_tracks(self, f)

    @staticmethod
    def load(filename=None, f=None):
        """Construct by loading from HDF5 file"""
        if f is None:
            with h5py.File(filename, 'r') as f:
                tracks_cube, centroids, pitch, rotations, n_tracks = _load_tracks(f)
        else:
            tracks_cube, centroids, pitch, rotations, n_tracks = _load_tracks(f)
        return DenseSquareTracks(tracks_cube, centroids, pitch, rotations)

    @staticmethod
    def combine(tracks):
        """Construct by combining from list of DenseSquareTracks"""
        # Assume all sub-tracks are the same dimensions
        n_tracks = np.sum([track.n_tracks for track in tracks])
        dims = tracks[0].tracks_cube.shape[1:]  # takes the 2nd+3rd dims
        tracks_cube = np.zeros((n_tracks, dims[0], dims[1]), dtype=np.int16)
        centroids = np.zeros((n_tracks, 2), dtype=np.float32)
        pitch = tracks[0].pitch
        rotations = np.zeros(n_tracks)
        i_start = 0
        for track in tracks:
            n_track = track.n_tracks
            i_stop = i_start + n_track
            tracks_cube[i_start:i_stop, :, :] = track.tracks_cube
            centroids[i_start:i_stop, :] = track.centroids
            rotations[i_start:i_stop] = track.rotations
            i_start = i_stop
        return DenseSquareTracks(tracks_cube, centroids, pitch, rotations)


class DenseSquareSimTracks(DenseSquareTracks):
    """Struct of arrays form of multiple tracks stored as arrays of images with simulated data."""

    def __init__(self, tracks_cube, centroids, pitch, rotations, angles, absorption_points):
        super().__init__(tracks_cube, centroids, pitch, rotations)
        self.angles = angles
        self.absorption_points = absorption_points
        n_pixels = tracks_cube.shape[2]
        self.tracks_max = np.max(np.reshape(tracks_cube,[-1,n_pixels,n_pixels]).astype(np.float32),axis=0)
        self.tracks_mean = np.mean(np.divide(np.reshape(tracks_cube,[-1,n_pixels,n_pixels]).astype(np.float32), self.tracks_max, out=np.zeros_like(np.reshape(tracks_cube,[-1,n_pixels,n_pixels]).astype(np.float32)), where=self.tracks_max!=0), axis=0) #np.mean(tracks_cube.astype(np.float32) / self.tracks_max, axis=0)
        self.tracks_std = np.std(np.divide(np.reshape(tracks_cube,[-1,n_pixels,n_pixels]).astype(np.float32), self.tracks_max, out=np.zeros_like(np.reshape(tracks_cube,[-1,n_pixels,n_pixels]).astype(np.float32)), where=self.tracks_max!=0), axis=0) #np.std(tracks_cube.astype(np.float32) / self.tracks_max, axis=0)

    def save(self, filename=None, f=None):
        """Save to HDF5 file. Either specify a filename to save a single object or f to save multiple objects"""
        if f is None:
            with h5py.File(filename, 'w') as f:
                _save_tracks(self, f)
                _save_tracks_sim(self, f)
        else:
            _save_tracks(self, f)
            _save_tracks_sim(self, f)

    @staticmethod
    def load(filename=None, f=None):
        """Construct by loading from HDF5 file"""
        if f is None:
            with h5py.File(filename, 'r') as f:
                tracks_cube, centroids, pitch, rotations, n_tracks, tracks_max, tracks_mean, tracks_std = _load_tracks(f)
                angles, absorption_points = _load_tracks_sim(f)
        else:
            tracks_cube, centroids, pitch, rotations, n_tracks, tracks_max, tracks_mean, tracks_std = _load_tracks(f)
            angles, absorption_points = _load_tracks_sim(f)
        return DenseSquareSimTracks(tracks_cube, centroids, pitch, rotations, angles, absorption_points)

    @staticmethod
    def combine(tracks):
        """Construct by combining from list of DenseSquareSimTracks"""
        # Assume all sub-tracks are the same dimensions
        n_tracks = np.sum([track.n_tracks for track in tracks])
        dims = tracks[0].tracks_cube.shape[1:]  # takes the 2nd+3rd dims
        tracks_cube = np.zeros((n_tracks, dims[0], dims[1]), dtype=np.int16)
        centroids = np.zeros((n_tracks, 2), dtype=np.float32)
        pitch = tracks[0].pitch
        rotations = np.zeros(n_tracks)
        angles = np.zeros((n_tracks,), dtype=np.float32)
        absorption_points = np.zeros((n_tracks, 3), dtype=np.float32)
        i_start = 0
        for track in tracks:
            n_track = track.n_tracks
            i_stop = i_start + n_track
            tracks_cube[i_start:i_stop, :, :] = track.tracks_cube
            centroids[i_start:i_stop, :] = track.centroids
            rotations[i_start:i_stop] = track.rotations
            angles[i_start:i_stop] = track.angles
            absorption_points[i_start:i_stop, :] = track.absorption_points
            i_start = i_stop
        return DenseSquareSimTracks(tracks_cube, centroids, pitch, rotations, angles, absorption_points)


def _save_tracks(tracks, f):
    """Helper function for saving tracks"""
    opts = {'compression': 'gzip'}
    f.create_dataset('tracks_cube', data=tracks.tracks_cube, **opts)
    f.create_dataset('centroids', data=tracks.centroids, **opts)
    f.create_dataset('pitch', data=tracks.pitch)
    f.create_dataset('rotations', data=tracks.rotations)
    f.create_dataset('n_tracks', data=tracks.n_tracks)
    f.create_dataset('tracks_max', data=tracks.tracks_max)
    f.create_dataset('tracks_mean', data=tracks.tracks_mean)
    f.create_dataset('tracks_std', data=tracks.tracks_std)
    f.attrs['class'] = 'DenseSquareTracks'


def _save_tracks_sim(tracks, f):
    """Helper function for saving additional sim fields"""
    opts = {'compression': 'gzip'}
    f.create_dataset('angles', data=tracks.angles, **opts)
    f.create_dataset('absorption_points', data=tracks.absorption_points, **opts)
    f.attrs['class'] = 'DenseSquareSimTracks'


def _load_tracks(f):
    """Helper function for loading tracks from open HDF5 file"""
    tracks_cube = f['tracks_cube'][...]
    centroids = f['centroids'][...]
    pitch = f['pitch'][...]
    rotations = f['rotations'][...]
    n_tracks = f['n_tracks'][...]
    tracks_max = f['tracks_max'][...]
    tracks_mean = f['tracks_mean'][...]
    tracks_std = f['tracks_std'][...]
    return tracks_cube, centroids, pitch, rotations, n_tracks, tracks_max, tracks_mean, tracks_std


def _load_tracks_sim(f):
    """Helper function for loading additional sim fields"""
    angles = f['angles'][...]
    absorption_points = f['absorption_points'][...]
    return angles, absorption_points
