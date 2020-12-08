import numpy as np
import h5py


class SparseHexTracks:
    """Struct of arrays form of multiple tracks converted to (x,y): Q form, with x and y the actual coordinates
    Performance note: it's fast to construct this from 3 lists of components, but would be slow to construct track-by-
    track. It's fast to get a single track from this, but would be slow to insert/delete tracks. Only the fast methods
    are implemented. Now includes the Moment analysis output parameters for each track.
    """

    def __init__(self, xs, ys, Qs, moms, mom_phis, mom_abs_pts, bars):
        """Construct by feeding lists of x,y,Q components - corresponding entries are the same track. xs, ys, and Qs
        can be lists of ndarrays or 2d ndarrays already.
        """
        n_tracks = len(Qs)
        assert len(xs) == n_tracks
        assert len(ys) == n_tracks

        # Handle construction from empty arrays as when loading
        if n_tracks == 0:
            self.x = None
            self.y = None
            self.Q = None
            self.mom = None
            self.mom_phi = None
            self.mom_abs_pt = None
            self.bar = None
            self.n_tracks = 0
            self.n_pixels = np.zeros((0,))
            return

        # Handle construction from 2d ndarrays already
        if type(xs) == np.ndarray and type(ys) == np.ndarray and type(Qs) == np.ndarray and type(moms) == np.ndarray and type(mom_phis) == np.ndarray:
            self.x = xs
            self.y = ys
            self.Q = Qs
            self.mom = moms
            self.mom_phi = mom_phis
            self.mom_abs_pt = mom_abs_pts
            self.bar = bars
            self.n_tracks = xs.shape[0]
            # Extract the non-sentinel values to get n_pixels
            n_pixels = np.zeros((n_tracks,), dtype=np.int32)
            for i in range(self.n_tracks):
                Q = Qs[i, :]
                pos = np.where(Q < 0)[0]
                if pos.size == 0:  # takes up the entire width
                    n_pixels[i] = Q.size
                else:
                    n_pixels[i] = pos[0]
            self.n_pixels = n_pixels
            return

        # Calculate max size of the arrays according to the "biggest" track
        n_pixels = np.array([x.size for x in xs], dtype=np.int32)
        n_max_pixels = np.max(n_pixels)

        # Initialize the array that will hold everything
        x = np.zeros((n_tracks, n_max_pixels), dtype=np.float32)  # sentinel value = NaN
        x[:] = np.nan
        y = np.zeros((n_tracks, n_max_pixels), dtype=np.float32)  # sentinel value = NaN
        y[:] = np.nan
        Q = np.zeros((n_tracks, n_max_pixels), dtype=np.int16)  # sentinel value = -1
        Q[:] = -1
        mom = np.zeros(n_tracks, dtype=np.float32)
        mom_phi = np.zeros(n_tracks, dtype=np.float32)
        mom_abs_pt = np.zeros((n_tracks, 2), dtype=np.float32)
        bar = np.zeros((n_tracks, 2), dtype=np.float32)

        # Fill in those arrays
        for i, (x_, y_, Q_, mom_, mom_phi_, mom_abs_pt_, bar_) in enumerate(zip(xs, ys, Qs, moms, mom_phis, mom_abs_pts, bars)):
            n = x_.size
            x[i, :n] = x_
            y[i, :n] = y_
            Q[i, :n] = Q_
            mom[i, :n] = mom_
            mom_phi[i, :n] = mom_phi_
            mom_abs_pt[i, :n] = mom_abs_pt_
            bar[i, :n] = bar_

        self.x = x
        self.y = y
        self.Q = Q
        self.mom = mom
        self.mom_phi = mom_phi
        self.mom_abs_pt = mom_abs_pt
        self.bar = bar

        # Save metadata for easier indexing/slicing ops
        self.n_tracks = n_tracks
        self.n_pixels = n_pixels

    def __getitem__(self, i):
        """Gets a single SparseHexTrack of the specified index i or a new SparseHexTracks of the specified indexes
        as an ndarray. This implements a __get__item implicit iterator.
        Note that this allows numpy's general array indexing.
        """
        # Multi-integer index returns a new struct of arrays
        if type(i) == np.ndarray:
            xs = self.x[i, :]
            ys = self.y[i, :]
            Qs = self.Q[i, :]
            moms = self.mom[i]
            mom_phis = self.mom_phi[i]
            mom_abs_pts = self.mom_abs_pt[i,:]
            bars = self.bar[i,:]
            return SparseHexTracks(xs, ys, Qs, moms, mom_phis, mom_abs_pts, bars)

        # Single integer index and iteration returns a single value
        if i == self.n_tracks:
            raise IndexError
        #n = self.n_pixels[i]
        x = self.x[i, :]
        y = self.y[i, :]
        Q = self.Q[i, :]
        mom = self.mom[i]
        mom_phi = self.mom_phi[i]
        mom_abs_pt = self.mom_abs_pt[i,:]
        bar = self.bar[i,:]
        return SparseHexTrack(x, y, Q, mom, mom_phi, mom_abs_pt, bar)

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
                x, y, Q, n_tracks, n_pixels = _load_tracks(f)
        else:
            x, y, Q, n_tracks, n_pixels = _load_tracks(f)
        return SparseHexTracks(x, y, Q)

    @staticmethod
    def combine(hex_tracks):
        """Construct by combining from list of SparseHexTrack or SparseHexTracks"""
        # From individual SparseHexTrack
        if isinstance(hex_tracks[0], SparseHexTrack):
            xs = []
            ys = []
            Qs = []
            for hex_track in hex_tracks:
                xs.append(hex_track.x)
                ys.append(hex_track.y)
                Qs.append(hex_track.Q)
            return SparseHexTracks(xs, ys, Qs)
        elif isinstance(hex_tracks[0], SparseHexTracks):
            # Directly work with ndarrays - faster than converting to individual tracks above
            n_tracks = np.sum([tracks.Q.shape[0] for tracks in hex_tracks])
            max_pixels = np.max([np.max(tracks.n_pixels) for tracks in hex_tracks])
            xs = np.zeros((n_tracks, max_pixels), dtype=np.float32)
            ys = np.zeros_like(xs)
            Qs = np.zeros((n_tracks, max_pixels), dtype=np.int16)
            xs[:] = np.nan
            ys[:] = np.nan
            Qs[:] = -1
            i_start = 0
            for tracks in hex_tracks:
                n_track, n_pixel = tracks.Q.shape
                i_stop = i_start + n_track
                xs[i_start:i_stop, :n_pixel] = tracks.x
                ys[i_start:i_stop, :n_pixel] = tracks.y
                Qs[i_start:i_stop, :n_pixel] = tracks.Q
                i_start = i_stop
            return SparseHexTracks(xs, ys, Qs)
        else:
            raise ValueError('Iterable of sparse hex track(s) not recognized')


class SparseHexTrack:
    """A single x,y,Q track in sparse coordinates"""

    def __init__(self, x, y, Q, mom, mom_phi, mom_abs_pt, bar):
        """Build from separate x, y, and Q arrays. Expects that sentinel values are not present."""
        n_pixels = Q.size
        assert x.size == n_pixels
        assert y.size == n_pixels

        self.x = x
        self.y = y
        self.Q = Q
        self.mom = mom
        self.mom_phi = mom_phi
        self.mom_abs_pt = mom_abs_pt
        self.bar = bar


class SparseHexSimTracks(SparseHexTracks):
    """SparseHexTracks augmented with simulation-only data."""

    def __init__(self, xs, ys, Qs, angles, absorption_points, moms, mom_phis, mom_abs_pts, bars, zs):
        """Construct augmented sparse hex tracks

        Args:
            angles: n_tracks ndarray of float32
            absorption_points: n_tracks x 3 ndarray of float32, each row is x,y,z
        """
        super().__init__(xs, ys, Qs, moms, mom_phis, mom_abs_pts, bars)

        assert angles.size == self.n_tracks
        self.angles = angles

        assert absorption_points.shape[0] == self.n_tracks
        self.absorption_points = absorption_points

        assert zs.shape[0] == self.n_tracks
        self.zs = zs

    def __getitem__(self, i):
        """Gets a single SparseHexSimTrack of the specified index i or a new SparseHexSimTracks of the specified indexes
        as an ndarray. This implements a __get__item implicit iterator.
        Note that this allows numpy's general array indexing.
        """
        # Multi-integer index returns a new struct of arrays
        if type(i) == np.ndarray:
            xs = self.x[i, :]
            ys = self.y[i, :]
            Qs = self.Q[i, :]
            moms = self.mom[i]
            mom_phis = self.mom_phi[i]
            mom_abs_pts = self.mom_abs_pt[i,:]
            bars = self.bar[i,:]
            angles = self.angles[i]
            absorption_points = self.absorption_points[i, :]
            zs = self.zs[i]
            return SparseHexSimTracks(xs, ys, Qs, angles, absorption_points, moms, mom_phis, mom_abs_pts, bars, zs)

        # Single integer index and iteration returns a single value
        if i == self.n_tracks:
            raise IndexError
        n = self.n_pixels[i]
        x = self.x[i, :n]
        y = self.y[i, :n]
        Q = self.Q[i, :n]
        return SparseHexSimTrack(x, y, Q, self.angles[i], self.absorption_points[i, :], self.mom[i], self.mom_phi[i], self.mom_abs_pt[i,:], self.bar[i,:], self.zs[i])

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
                x, y, Q, n_tracks, n_pixels = _load_tracks(f)
                angles, absorption_points, zs = _load_tracks_sim(f)
        else:
            x, y, Q, n_tracks, n_pixels = _load_tracks(f)
            angles, absorption_points, zs = _load_tracks_sim(f)
        return SparseHexSimTracks(x, y, Q, angles, absorption_points, zs)

    @staticmethod
    def combine(hex_tracks):
        """Construct by combining from list of SparseHexSimTrack or SparseHexSimTracks"""
        # From individual SparseHexTrack
        if isinstance(hex_tracks[0], SparseHexSimTrack):
            xs = []
            ys = []
            Qs = []
            angles = []
            zs = []
            absorption_points = []
            for hex_track in hex_tracks:
                xs.append(hex_track.x)
                ys.append(hex_track.y)
                Qs.append(hex_track.Q)
                angles.append(hex_track.angle)
                zs.append(hex_track.z)
                absorption_points.append(hex_track.absorption_point)
            angles = np.array(angles)
            zs = np.array(zs)
            absorption_points = np.array(absorption_points)
            return SparseHexSimTracks(xs, ys, Qs, angles, absorption_points, zs)
        elif isinstance(hex_tracks[0], SparseHexSimTracks):
            # Directly work with ndarrays - faster than converting to individual tracks above
            n_tracks = np.sum([tracks.Q.shape[0] for tracks in hex_tracks])
            max_pixels = np.max([np.max(tracks.n_pixels) for tracks in hex_tracks])
            xs = np.zeros((n_tracks, max_pixels), dtype=np.float32)
            ys = np.zeros_like(xs)
            Qs = np.zeros((n_tracks, max_pixels), dtype=np.int16)
            angles = np.zeros((n_tracks,), dtype=np.float32)
            zs = np.zeros((n_tracks,), dtype=np.float32)
            absorption_points = np.zeros((n_tracks, 3), dtype=np.float32)
            xs[:] = np.nan
            ys[:] = np.nan
            Qs[:] = -1
            i_start = 0
            for tracks in hex_tracks:
                n_track, n_pixel = tracks.Q.shape
                i_stop = i_start + n_track
                xs[i_start:i_stop, :n_pixel] = tracks.x
                ys[i_start:i_stop, :n_pixel] = tracks.y
                Qs[i_start:i_stop, :n_pixel] = tracks.Q
                angles[i_start:i_stop] = tracks.angles
                zs[i_start:i_stop] = tracks.zs
                absorption_points[i_start:i_stop, :] = tracks.absorption_points
                i_start = i_stop
            return SparseHexSimTracks(xs, ys, Qs, angles, absorption_points)
        else:
            raise ValueError('Iterable of sparse hex sim track(s) not recognized')


class SparseHexSimTrack(SparseHexTrack):
    """SparseHexTrack augmented with simulation-only data."""

    def __init__(self, x, y, Q, angle, absorption_point, mom, mom_phi, mom_abs_pt, bar, z):
        super().__init__(x, y, Q, mom, mom_phi, mom_abs_pt, bar)

        self.angle = angle
        self.absorption_point = absorption_point
        self.z = z


def _save_tracks(tracks, f):
    """Helper function for saving tracks from open HDF5 file"""
    opts = {'compression': 'gzip'}
    f.create_dataset('x', data=tracks.x, **opts)
    f.create_dataset('y', data=tracks.y, **opts)
    f.create_dataset('Q', data=tracks.Q, **opts)
    f.create_dataset('n_tracks', data=tracks.n_tracks)
    f.create_dataset('n_pixels', data=tracks.n_pixels, **opts)
    f.attrs['class'] = 'SparseHexTracks'


def _save_tracks_sim(tracks, f):
    """Helper function for saving additional sim fields"""
    opts = {'compression': 'gzip'}
    f.create_dataset('angles', data=tracks.angles, **opts)
    f.create_dataset('zs', data=tracks.zs, **opts)
    f.create_dataset('absorption_points', data=tracks.absorption_points, **opts)
    f.attrs['class'] = 'SparseHexSimTracks'


def _load_tracks(f):
    """Helper function for loading tracks from open HDF5 file"""
    x = f['x'][...]
    y = f['y'][...]
    Q = f['Q'][...]
    n_tracks = f['n_tracks'][...]
    n_pixels = f['n_pixels'][...]
    return x, y, Q, n_tracks, n_pixels


def _load_tracks_sim(f):
    """Helper function for loading additional sim fields"""
    angles = f['angles'][...]
    zs = f['zs'][...]
    absorption_points = f['absorption_points'][...]
    return angles, absorption_points, zs
