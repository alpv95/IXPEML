import numpy as np


class RoiHexTracks:
    """Struct of arrays form of original compressed hex data in ROI
    Performance note: it's fast to construct this from a list of tracks+rois, but would be slow to construct track-by-
    track. It's fast to get a single track from this, but would be slow to insert/delete tracks. Only the fast methods
    are implemented.
    """

    def __init__(self, tracks, rois):
        """Construct by feeding in all track+roi data, and optionally angles and absorption points for sims.
        Can also construct by feeding in ndarrays of everything.

        Args:
            tracks: n_tracks list of ndarray of ints, the raw pixel data in a linear format that's specified by the ROI
            rois: n_tracks x 4 ndarray of int32, the ROIs where each row is (min_col, max_col, min_row, max_row)
            angles: n_tracks ndarray of int32, the track starting angle for sims
            absorption_points: n_tracks x 3 ndarray of int32, the absorption point x,y,z coords for sims
        """
        # Handle construction from ndarrays
        if type(tracks) == np.ndarray and type(rois) == np.ndarray:
            self.tracks_arr = tracks
            self.rois = rois
            self.n_tracks = tracks.shape[0]
            return

        # Get biggest track size (ROI) to allow storage as a single 2-D ndarray
        track_pixels = np.array([track.size for track in tracks], dtype=np.int32)
        max_track_pixels = np.max(track_pixels)

        # Store all the tracks in a single 2-D ndarray
        n_tracks = len(tracks)
        tracks_arr = np.ones((n_tracks, max_track_pixels), dtype=np.int16) * -1  # -1 is a sentinel/invalid val
        for i, track in enumerate(tracks):
            n_pixels = track.size
            tracks_arr[i, :n_pixels] = track

        # Store main data
        self.tracks_arr = tracks_arr
        self.rois = rois

        # Store metadata
        self.n_tracks = len(rois)

    def __getitem__(self, i):
        """Gets a single RoiHexTrack of the specified index i or a new RoiHexTracks of the specified indexes
        as an ndarray. This implements a __get__item implicit iterator.
        Note that this allows numpy's general array indexing.
        """
        # Multi-integer index returns a new struct of arrays
        if type(i) == np.ndarray:
            return RoiHexTracks(self.tracks_arr[i, :], self.rois[i])

        # Single integer index and iteration returns a single value
        if i == self.n_tracks:
            raise IndexError
        return RoiHexTrack(self.tracks_arr[i, :], self.rois[i])


class RoiHexTrack:
    """A single original compressed hex data in ROI"""

    def __init__(self, track, roi):
        """Build from a linear track array and roi specifier. Cleans up sentinel -1 vals in track and checks
        consistency with roi.
        """
        min_col, max_col, min_row, max_row = roi
        roi_size = (max_col - min_col + 1) * (max_row - min_row + 1)
        self.track = track[:roi_size]
        self.roi = roi


class RoiSimHexTracks(RoiHexTracks):
    """RoiSimHexTracks augmented with simulation-only data."""

    def __init__(self, tracks, rois, angles, absorption_points):
        super().__init__(tracks, rois)

        assert angles.size == self.n_tracks
        self.angles = angles

        assert absorption_points.shape[0] == self.n_tracks
        self.absorption_points = absorption_points

    def __getitem__(self, i):
        # Multi-integer index returns a new struct of arrays
        if type(i) == np.ndarray:
            return RoiSimHexTracks(self.tracks_arr[i, :], self.rois[i], self.angles[i], self.absorption_points[i, :])

        # Single integer index and iteration returns a single value
        if i == self.n_tracks:
            raise IndexError
        return RoiSimHexTrack(self.tracks_arr[i, :], self.rois[i], self.angles[i], self.absorption_points[i, :])


class RoiSimHexTrack(RoiHexTrack):
    """RoiSimHexTrack augmented with simulation-only data"""

    def __init__(self, track, roi, angle, absorption_point):
        super().__init__(track, roi)

        self.angle = angle
        self.absorption_point = absorption_point
