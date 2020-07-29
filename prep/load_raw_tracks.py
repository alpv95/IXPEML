import numpy as np
import multiprocessing
from astropy.io import fits
from formats.roi_hex import RoiHexTracks, RoiSimHexTracks
from formats.sparse_hex import SparseHexTracks, SparseHexSimTracks


def load_hex_fits(input_file, data_type='meas', n_tracks_limit=None, sim_hdu=2):
    """Load tracks data in sequential hex + ROI form from .fits file.

    Note: Uses memmap=False when opening the .fits file. So the whole file is loaded all at once (and assumes it
    can fit in memory).

    Args:
        input_file: str, location of .fits file containing raw data
        data_type: str, 'meas' or 'sim', where the dataset came from. If 'sim', additional fields for angle and
            absorption point are saved.
        n_tracks_limit: int or None, How many tracks to get, or None=all of them.
        sim_hdu: int, For data_type='sim', which hdu index contains the ground truth/path data. It was in hdu1 for the
            original dataset and hdu3 for set2.

    Returns:
        RoiHexTracks, object storing all the raw hex tracks
    """
    if data_type not in {'meas', 'sim'}:
        raise ValueError('Unknown data type')

    with fits.open(input_file, memmap=False) as hdu:
        # print(hdu[1].columns)
        data = hdu[1].data

        n_tracks = data.size
        if n_tracks_limit is not None:
            n_tracks = n_tracks_limit

        # Get metadata
#        min_cols = data['MIN_COL'][:n_tracks]
#        max_cols = data['MAX_COL'][:n_tracks]
#        min_rows = data['MIN_ROW'][:n_tracks]
#        max_rows = data['MAX_ROW'][:n_tracks]
#        rois = np.column_stack((min_cols, max_cols, min_rows, max_rows))

        # Get data
        #   Note: This is slow since it's a jagged array of arrays
        #   These are stored as int16
#        tracks = [data['PIX_PHAS'][i] for i in range(n_tracks)]

        tracks_x = [data['PIX_X'][i] for i in range(n_tracks)]
        tracks_y = [data['PIX_Y'][i] for i in range(n_tracks)]
        tracks_inv = [data['PIX_INV'][i] for i in range(n_tracks)]
        tracks_adc = [data['PIX_PHA'][i] for i in range(n_tracks)]
        tracks_mom_angles = np.array([data['DETPHI'][i] for i in range(n_tracks)])
        tracks_mom_abs_x = np.array([data['DETX'][i] for i in range(n_tracks)])
        tracks_mom_abs_y = np.array([data['DETY'][i] for i in range(n_tracks)])
        tracks_mom_abs_pts = np.column_stack((tracks_mom_abs_x, tracks_mom_abs_y))
        tracks_mom = np.array([data['TRK_M2L'][i] / data['TRK_M2T'][i] for i in range(n_tracks)])

        # Get ground truth data
        if data_type == 'sim':
            sims = hdu[sim_hdu].data
            angles = sims['PE_PHI'][:n_tracks]
            abs_xs = sims['ABS_X'][:n_tracks]
            abs_ys = sims['ABS_Y'][:n_tracks]
            abs_zs = sims['ABS_Z'][:n_tracks]
            absorption_points = np.column_stack((abs_xs, abs_ys, abs_zs))
        else:
            angles = None
            absorption_points = None

    # Get distribution of n_pixels among tracks
    #   There's a fairly narrow range of vals - OK to just pack everything into 1 matrix
    # plt.figure()
    # plt.hist(track_lengths)
    # plt.xlabel('# Pixels')
    # plt.ylabel('Count')
    # plt.title('Distribution of # Pixels')
    # plt.show()
    if data_type == 'sim':
       hex_tracks = SparseSimHexTracks(tracks_x, tracks_y, tracks_adc, angles, absorption_points, tracks_mom, tracks_mom_angles, tracks_mom_abs_pts)
    else:
       hex_tracks = SparseHexTracks(tracks_x, tracks_y, tracks_adc, tracks_mom, tracks_mom_angles, tracks_mom_abs_pts)

    # Build struct of arrays object to hold all raw track data
#    if data_type == 'sim':
#        raw_tracks = RoiSimHexTracks(tracks, rois, angles, absorption_points)
#    else:
#        raw_tracks = RoiHexTracks(tracks, rois)
#    return raw_tracks
    return hex_tracks

# Hex grid params
grid_x_0 = -7.4875  # (mm), counts up
grid_x = 2 * 7.4875  # (mm)
grid_px = 300 - 0.5
pixel_x = grid_x / grid_px  # (mm), pixel width
grid_y_0 = 7.5991  # (mm), counts down
grid_y = 2 * 7.5991  # (mm)
grid_py = 352 - 1
pixel_y = grid_y / grid_py  # (mm), pixel height


def convert_hex_tracks(raw_tracks, pixel_cutoff=None, compact=True, n_cores=1):
    """Convert ROI hex tracks (original compressed linear form) -> sparse hex tracks (actual coordinates).
    Note that this function currently stores all the parameters of the original compressed linear form - these could
    be moved to the .fits file loader and stored with the RoiHexTracks object.

    Args:
        raw_tracks: RoiHexTracks, object storing all the raw hex tracks
        pixel_cutoff: int or None, A value below which a pixel's value is reported as 0, or None
            for no cutoff. (Keeps the pixel_cutoff value or higher)
        compact: bool, Whether to remove 0 pixels. If True, saves space in memory/storage but requires a more expensive
            step for interpolation. The opposite for False. Default is True.
        n_cores: int, Number of cores to run in parallel. Default is 1.

    Returns:
        SparseHexTracks, object storing all the sparse hex tracks
    """
    # Split tracks among cores
    n_tracks = raw_tracks.n_tracks
    splits = np.array_split(np.arange(n_tracks), n_cores)

    # Run in parallel
    #with multiprocessing.Pool(processes=n_cores) as pool:
        #results = pool.starmap(convert_hex_tracks_sub, [(raw_tracks[split], pixel_cutoff, compact) for split in splits])

    #Run one process only
    results = convert_hex_tracks_sub(raw_tracks, pixel_cutoff, compact)

    # Combine
    if hasattr(raw_tracks, 'angles') and hasattr(raw_tracks, 'absorption_points'):
        hex_tracks = SparseHexSimTracks.combine(results)
    else:
        hex_tracks = SparseHexTracks.combine(results)

    return hex_tracks


def convert_hex_tracks_sub(raw_tracks, pixel_cutoff=None, compact=True):
    """Subprocess for converting hex tracks for running in parallel. Same args as convert_hex_tracks except n_cores."""
    # Make lists to store all the sparse hex tracks components
    xs_all = []
    ys_all = []
    Qs_all = []
    for i_track, track in enumerate(raw_tracks):
        if i_track % 1000 == 0:
            print('Converting track {}'.format(i_track))

        track_hex_1d = track.track

        # Apply cutoff
        if pixel_cutoff is not None:
            track_hex_1d[track_hex_1d < pixel_cutoff] = 0

        min_col, max_col, min_row, max_row = track.roi
        n_cols = max_col - min_col + 1
        n_rows = max_row - min_row + 1

        # Convert track data into 2-D hex
        track_hex_2d = track_hex_1d.reshape((n_rows, n_cols))

        # Get cartesian indexes for each pt
        row_inds = np.arange(min_row, max_row + 1)
        col_inds = np.arange(min_col, max_col + 1)

        # Row locations are all consistent
        y_hex = grid_y_0 - (row_inds * pixel_y)
        y_hexs = np.tile(y_hex.reshape(-1, 1), (1, n_cols)).astype(np.float32)

        # Col locations are offset every other row.
        #   Odds are flush, evens are offset
        x_hex_odd = grid_x_0 + (col_inds * pixel_x)
        x_hex_even = grid_x_0 + ((col_inds + 0.5) * pixel_x)
        # Fill in starting with right phase
        #   Note: It actually looks like all ROIs start with an even col, but this accounts for possible odd col start
        x_hexs = np.zeros_like(track_hex_2d, dtype=np.float32)
        if min_col % 2 == 0:
            x_hexs[::2, :] = x_hex_even
            x_hexs[1::2, :] = x_hex_odd
        else:
            x_hexs[::2, :] = x_hex_odd
            x_hexs[1::2, :] = x_hex_even

        # Convert to sparse form
        xs = x_hexs.reshape(-1)
        ys = y_hexs.reshape(-1)
        Qs = track_hex_2d.reshape(-1)

        # Dev/test - plot points
        # plt.figure()
        # plt.scatter(xs, ys, Qs)
        # plt.axis('equal')
        # plt.title('Hex {}'.format(i_track))
        # plt.show()

        # Optionally compact away 0's
        if compact:
            keep = Qs > 0
            xs = xs[keep]
            ys = ys[keep]
            Qs = Qs[keep]

        # Save tracks
        xs_all.append(xs)
        ys_all.append(ys)
        Qs_all.append(Qs)

    # Combine and save as struct of arrays
    if hasattr(raw_tracks, 'angles') and hasattr(raw_tracks, 'absorption_points'):
        hex_tracks = SparseHexSimTracks(xs_all, ys_all, Qs_all, raw_tracks.angles, raw_tracks.absorption_points)
    else:
        hex_tracks = SparseHexTracks(xs_all, ys_all, Qs_all)
    return hex_tracks
