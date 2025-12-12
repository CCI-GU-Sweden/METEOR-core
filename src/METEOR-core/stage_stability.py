# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 08:49:58 2025

@author: simon

Requires to have in the name of the file the step size used, e.g.:
- "0.125" for 0.125 micron step
- "0.25" for 0.25 micron step
- if not found, a default of 1 micron is used.
This script measures the stage stability (drift) in XY and Z directions
over a time-lapse acquisition of a stair-step calibration sample.

Metadata must contain:
- 'Physical pixel size X' : pixel size in micron
- 'Physical pixel size Y' : pixel size in micron
- 'Lens Magnification' : magnification of the objective lens
- 'Image Size Z' : number of Z slices - should be 1

Warning! If not found, a default step size and physical pixel size of 1 micron is used.

"""
#general import
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from skimage import measure, transform
from scipy import ndimage
import re

#local import
from functions import reader, norm_img, rings_detection, dimension_slicer, aggregate_close_points
from stack_accurancy import remove_corners
import logger

#global parameters
CORNER_TOL_UM = 2
RING_RADIUS = .6 #in micrometer
ROW_BAND_UM = 3
step = 3.6
magic_f = 2
first_row_um = 14.5
second_row_um = 7.0
ring_by_stair = 11

def detect_step_from_filename(filename:str):
    """
    Detect the step size from the filename.
    """
    # Known step sizes in microns
    known_steps = [1.0, 0.5, 0.25, 0.125]

    # Get basename only
    fname = os.path.basename(filename).lower()
    fname = fname.replace(",", ".")  # handle comma decimal

    # Regex to find numbers like 1, 0.5, 0.25, 0.125
    matches = re.findall(r"[\d]*\.?[\d]+", fname)

    # Convert to float safely
    candidates = []
    for m in matches:
        try:
            val = float(m)
            candidates.append(val)
        except ValueError:
            pass

    # Pick candidate if it matches one of the known steps
    for c in candidates:
        for s in known_steps:
            if abs(c - s) < 1e-6:  # exact match
                return s

    # Fallback: pick closest if approx typo (like 0.124 instead of 0.125)
    for c in candidates:
        closest = min(known_steps, key=lambda s: abs(s-c))
        if abs(closest - c) < 0.1:  # tolerance in microns
            return closest

    return None  # not found



def get_rings_data(img:np.array, metadata:dict) -> tuple:
    """
    Extract per-ring data from a single 2D frame.

    Steps
    -----
    1. Detect rings using `rings_detection`.
    2. Find the four corner pillars.
    3. Remove rings close to corners.
    4. Keep only rings belonging to the two 'stair' rows.
    5. Cluster close centroids to reduce redundancy.
    6. For each ring:
        - assign stair index (0 or 1),
        - estimate Z-level (index along X, via global `step`),
        - compute ring contrast via polar transform (Michelson).

    Parameters
    ----------
    img : np.ndarray
        2D image (Y, X), normalized or raw.
    metadata : dict
        Must contain 'Physical pixel size X' in µm.

    Returns
    -------
    rings_data : list[dict] or None
        Each entry has keys: 'stair_idx', 'centroid', 'Z-level', 'contrast'.
        Returns None if detection fails.
    corner_points_micron : np.ndarray or None
        (4, 2) array of corner positions in µm (y, x),
        or None if detection fails.
    """
    try:
        pxl_size_xy = float(metadata.get('Physical pixel size X', 1.0))
    except (TypeError, ValueError):
        logger.error("[get_rings_data] Invalid 'Physical pixel size X' in metadata.")
        return None, None
    
    binary = rings_detection(img, metadata)
    labeled_img = measure.label(binary)
    regions = measure.regionprops(labeled_img, intensity_image=img)

    if not regions:
        logger.warning("[get_rings_data] No regions detected in frame.")
        return None, None

    centroid = [x.centroid_weighted for x in regions]
    centroid = np.array(centroid)

    centroid_micron = np.copy(centroid)
    centroid_micron *= pxl_size_xy
    
    #detect the corners
    ymin, xmin = centroid_micron.min(axis=0)
    ymax, xmax = centroid_micron.max(axis=0)

    corner_targets = np.array([
        [ymin, xmin],  # top-left
        [ymin, xmax],  # top-right
        [ymax, xmin],  # bottom-left
        [ymax, xmax]   # bottom-right
    ])
    
    corner_indices = []
    for ct in corner_targets:
        dists = np.linalg.norm(centroid_micron - ct, axis=1)
        corner_indices.append(np.argmin(dists))
    
    corner_points_micron = centroid_micron[corner_indices]

    #remove the corners
    centroid_micron_filtered = remove_corners(centroid_micron, corner_points_micron, tol_um=CORNER_TOL_UM)

    if centroid_micron_filtered.size == 0:
        logger.warning("[get_rings_data] All points removed near corners.")
        return None, corner_points_micron

    # Detect the stair rings
    top_y = (corner_points_micron[0,0] + corner_points_micron[1,0]) / 2
    row1_y = top_y + first_row_um     # first stair row
    row2_y = row1_y + second_row_um     # second stair row
    rows_y = [row1_y, row2_y]
    
    centroid_micron_filtered = np.array([
        coo for coo in centroid_micron_filtered
        if any(abs(coo[0] - row_y) < ROW_BAND_UM for row_y in rows_y)
    ])

    if centroid_micron_filtered.size == 0:
        logger.warning("[get_rings_data] No points near stair rows.")
        return None, corner_points_micron

    # cluster close rings in 2D (µm)
    centroid_micron_filtered, _ = aggregate_close_points(
        centroid_micron_filtered,
        diameter_um=2.0,
        return_groups=False,
        twoD=True,
    )

    if centroid_micron_filtered.size == 0:
        logger.warning("[get_rings_data] No points after clustering.")
        return None, corner_points_micron

    # Horizontal reference (X direction)
    top_x = (corner_points_micron[0,1] + corner_points_micron[2,1]) / 2  
    crop_size = int(RING_RADIUS/pxl_size_xy * magic_f)  #radius, in pxl

    rings_data = []
    H, W = img.shape
    for centroid in centroid_micron_filtered:
        ring_data = {}
        if centroid[0] > row1_y-ROW_BAND_UM and centroid[0] < row1_y+ROW_BAND_UM: #stair 1
            ring_data['stair_idx'] = 0
        elif centroid[0] > row2_y-ROW_BAND_UM and centroid[0] < row2_y+ROW_BAND_UM: #stair 2
            ring_data['stair_idx'] = 1
        else:
            continue
        ring_data['centroid'] = np.array(centroid)
        ring_data['Z-level'] = int(round((centroid[1]-top_x) / step))
        
        # crop ring in pixel coordinates with bounds checking
        if (int(centroid[0]/pxl_size_xy)-crop_size < 0 or
            int(centroid[0]/pxl_size_xy)+crop_size > H or
            int(centroid[1]/pxl_size_xy)-crop_size < 0 or
            int(centroid[1]/pxl_size_xy)+crop_size > W):
            logger.warning("[get_rings_data] Ring at edge, skipping contrast calculation.")
            continue

        ring = img[int(centroid[0]/pxl_size_xy)-crop_size:int(centroid[0]/pxl_size_xy)+crop_size,
                   int(centroid[1]/pxl_size_xy)-crop_size:int(centroid[1]/pxl_size_xy)+crop_size]
        
        if ring.size == 0:
            continue

        ring = norm_img(ring)
        cy, cx = ring.shape[0] // 2, ring.shape[1] // 2

        if cy <= 0 or cx <= 0:
            continue

        # Contrast extraction
        polar = transform.warp_polar(ring, center=(cy, cx), radius=cy)
        radial_profile = np.mean(polar, axis=0)
        radial_profile = ndimage.uniform_filter1d(radial_profile, size=5)   

        if radial_profile.size == 0:
            continue

        baseline = radial_profile[0]
        peak = np.max(radial_profile)

        if (peak + baseline) == 0:
            continue

        contrast = (peak - baseline) / (peak + baseline)
        ring_data['contrast'] = contrast
        
        #add the data to the result list
        rings_data.append(ring_data)

    if not rings_data:
        logger.warning("[get_rings_data] No valid rings extracted.")
        return None, corner_points_micron

    return rings_data, corner_points_micron


def drift_xy_analysis(corner_list:list, debug:bool=False) -> tuple:
    #XY Drift analysis using the 4 main pillars
    drift_xy = []
    for idx in range(4):
        coords = np.array([x[idx] for x in corner_list])
        drift_xy.append(np.diff(coords, axis=0))

    #average the 4 pillars relative position
    drift_xy = np.mean(drift_xy, axis=0)

    # Magnitude of each displacement
    step_distances = np.linalg.norm(drift_xy, axis=1)
    
    # Metrics
    mean_step = np.mean(step_distances)
    net_drift = np.linalg.norm(drift_xy[-1] - drift_xy[0])   
    
    if debug:
        logger.info(f"Mean XY frame-to-frame drift: {mean_step:.4f} µm")
        logger.info(f"Net XY drift (start to end): {net_drift:.4f} µm")
       
    return drift_xy, mean_step, net_drift

def drift_z_analysis(rings_list:list, step_micron:float, debug:bool=False) -> tuple:
    """
    Estimate Z drift from a list of per-frame ring data.

    For each frame:
      - pick the max-contrast ring on stair 0 and stair 1 (if available),
      - store their Z-level index.

    Then:
      - convert indices to microns using `step_micron`,
      - compute the mean Z position (averaging the two stairs),
      - compute:
          * mean_zstep: average |ΔZ| per frame,
          * net_zdrift: |Z_last - Z_first|.

    Parameters
    ----------
    rings_list : list[list[dict] or None]
        One entry per timepoint, each either None or a list of ring dicts
        as returned by `get_rings_data`.
    step_micron : float
        Nominal Z step in microns (from filename).
    debug : bool
        If True, log details.

    Returns
    -------
    focus_z_stair0 : np.ndarray
        Z indices of selected rings on stair 0 (may be shorter than T).
    focus_z_stair1 : np.ndarray
        Z indices of selected rings on stair 1 (may be shorter than T).
    mean_zstep : float or np.nan
        Mean absolute per-frame drift (µm).
    net_zdrift : float or np.nan
        Drift from first to last frame (µm).
    """
    focus_z_stair0, focus_z_stair1 = [], []
    
    for rings_data in rings_list:
        if not rings_data:
            continue

        for stair_idx in (0,1):
            stair_rings = [r for r in rings_data if r['stair_idx']==stair_idx]
            if not stair_rings:
                continue
            
            best_ring = max(stair_rings, key=lambda r: r.get('contrast', 0))
            z_best = best_ring.get('Z-level', None)
            if z_best is None:
                continue
            
            if stair_idx == 0:
                focus_z_stair0.append(z_best)
            else:
                # Argolight stair pattern flips indexing for the second row
                focus_z_stair1.append(abs(ring_by_stair-z_best))
    
    focus_z_stair0 = np.asarray(focus_z_stair0, dtype=float)
    focus_z_stair1 = np.asarray(focus_z_stair1, dtype=float)

    if focus_z_stair0.size == 0 and focus_z_stair1.size == 0:
        logger.warning("[drift_z_analysis] Not enough valid rings for one or both stairs.")
        return focus_z_stair0, focus_z_stair1, np.nan, np.nan
    elif focus_z_stair0.size == 0:
        z_mean = focus_z_stair1 * step_micron
        mean_zstep = np.abs(focus_z_stair1)
    elif focus_z_stair1.size == 0:
        z_mean = focus_z_stair0 * step_micron
        mean_zstep = np.abs(focus_z_stair0)
    else:
        z_mean = np.mean([focus_z_stair0, focus_z_stair1], axis=0) * step_micron
        mean_zstep = np.mean(np.abs(np.diff(z_mean)))               # typical per-frame drift magnitude
    
    net_zdrift = abs(z_mean[-1] - z_mean[0])         # displacement from start to end

    if debug:
        logger.info(f"Mean Z frame-to-frame drift: {mean_zstep:.4f} µm")
        logger.info(f"Net Z drift (start to end): {net_zdrift:.4f} µm")
       
    return np.array(focus_z_stair0), np.array(focus_z_stair1), mean_zstep, net_zdrift
    

def measure_stage_stability(img_array:np.ndarray, metadata:dict, visu:bool=True, debug:bool=False, dim:str='TYX') -> dict:
    """
    Measure stage stability (drift) in XY and Z over a time-lapse of a stair-step sample.

    Workflow
    --------
    1. Infer the nominal Z-step in microns from the filename (if possible).
    2. Normalize the stack (`norm_img`).
    3. For each timepoint (2D frame):
        - Extract ring/stair data and corner positions (`get_rings_data`).
    4. XY drift:
        - track each of the 4 corner pillars over time,
        - compute per-frame displacement (µm) and average across pillars,
        - compute:
            * mean_step: mean |ΔXY| per frame,
            * net_drift: |(ΔXY_last - ΔXY_first)|.
    5. Z drift:
        - for each timepoint, choose best-contrast ring on each stair,
        - compute mean Z position and its frame-to-frame variations.

    Parameters
    ----------
    img_array : np.ndarray
        Time-lapse stack over the *time* dimension.
        Expected shape: (T, Y, X) or (T, Z, Y, X); here you pass `img_array[:,0,...]`
        in your __main__ so we assume (T, Y, X).
    metadata : dict
        Must contain at least 'Physical pixel size X'.
        Optionally: 'filepath' for step detection from filename.
    visu : bool
        If True, returns a matplotlib figure with:
          - sum-projection with corners overlay,
          - XY drift trajectory,
          - Z focus level over time.
    debug : bool
        If True, log more details.

    Returns
    -------
    dict
        {
          "figure": fig or None,
          "figure name": "stage_stability",
          "status": "success" or "error",
          "measurements": [
              Mean XY Drift, Net XY Drift, Mean Z Drift, Net Z Drift
          ],
          "extra_measurements": {}
        }
    """

    if int(metadata.get('Image Size C', 0)) > 1: #security, in case of multi-C, select the first one
        img_array = img_array[:,:1,...]
        if debug:
            logger.info("stack_precision: multi-C image, taking C=0 only, new shape: "+str(img_array.shape))

    if int(metadata.get('Image Size Z', 0)) > 1: #security, in case of multi-Z, select the first one
        img_array = img_array[:,:,:1,:,:]
        if debug:
            logger.info("stack_precision: multi-Z image, taking Z=0 only, new shape: "+str(img_array.shape))

    img_array = dimension_slicer(img_array, metadata, dim)

    img_array = np.asarray(img_array)
    if img_array.ndim != 3:
        raise ValueError(f"[measure_stage_stability] expects a 3D time stack (T, Y, X), got {img_array.shape}")

    pxl_size_xy = float(metadata.get('Physical pixel size X', 1.0))
    
    # Default step if not detected
    step_micron = 1.0
    filepath = metadata.get("filepath", None)
    if filepath:
        filename = os.path.basename(filepath)
        detected = detect_step_from_filename(filename)
        if detected is not None:
            step_micron = detected
    
    # norm image
    img_array = norm_img(img_array)
    
    # per-frame ring & corner extraction with error handling
    rings_list = []
    corner_list = []
    for t, frame in enumerate(img_array):
        try:
            rings_data, corners = get_rings_data(frame, metadata)
        except Exception as e:
            logger.warning(f"[measure_stage_stability] Error in get_rings_data at frame {t}: {e}")
            rings_data, corners = None, None

        if rings_data is None or corners is None:
            logger.warning(f"[measure_stage_stability] Skipping frame {t} (no valid ring/corner data).")
            continue

        rings_list.append(rings_data)
        corner_list.append(corners)

    if not corner_list or not rings_list:
        logger.error("[measure_stage_stability] No valid frames for drift analysis.")
        return {
            "figure": None,
            "figure name": 'stage_stability',
            "status": "error",
            "measurements": [
                {'category': 'Stage Stability', 'name': 'Mean XY Drift', 'value': None, 'unit': 'µm'},
                {'category': 'Stage Stability', 'name': 'Net XY Drift', 'value': None, 'unit': 'µm'},
                {'category': 'Stage Stability', 'name': 'Mean Z Drift', 'value': None, 'unit': 'µm'},
                {'category': 'Stage Stability', 'name': 'Net Z Drift', 'value': None, 'unit': 'µm'},
            ],
            "extra_measurements": {},
        }
    #XY drift
    drift_xy, mean_step, net_drift = drift_xy_analysis(corner_list)

    #Z drift
    focus_z_stair0, focus_z_stair1, mean_zstep, net_zdrift = drift_z_analysis(rings_list, step_micron)

    #figure
    fig = None
    if visu:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
        
        ax1.imshow(np.sum(img_array, axis=0))
        #display the corners position on the image
        for corner in corner_list:
            ax1.scatter(corner[:,1]/pxl_size_xy, corner[:,0]/pxl_size_xy,
                        s=25, marker='x', c='red', linewidths=.1)   
        ax1.axis('off')
        ax1.set_title('Sum of the timelapse image')
        
        if drift_xy.size >= 2:
            # Build segments between consecutive points
            points = np.array([drift_xy[:, 1], drift_xy[:, 0]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            t = np.arange(len(drift_xy))  # time index
            lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(0, len(drift_xy)-1))
            lc.set_array(t)
            lc.set_linewidth(2)
            ax2.add_collection(lc)
            ax2.autoscale()
            ax2.set_xlabel('ΔY (µm)')
            ax2.set_ylabel('ΔX (µm)')
            ax2.set_title(f"Average Stage Drift (XY) per Frame: {mean_step:.4f} µm")
            plt.colorbar(lc, ax=ax2, label='Timepoint')
        else:
            ax2.text(0.5, 0.5, "Not enough data for XY drift", ha='center', va='center')
            ax2.axis('off')
        
        # Z drift
        if focus_z_stair0.size > 0 or focus_z_stair1.size > 0:
            if focus_z_stair0.size > 0:
                ax3.plot(focus_z_stair0 * step_micron, '--', label="Stair 0")
            if focus_z_stair1.size > 0:
                ax3.plot(focus_z_stair1 * step_micron, '--', label="Stair 1")

            if focus_z_stair0.size > 0 and focus_z_stair1.size > 0:
                n = min(focus_z_stair0.size, focus_z_stair1.size)
                mean_curve = 0.5 * (focus_z_stair0[:n] + focus_z_stair1[:n]) * step_micron
                ax3.plot(mean_curve, label='Mean')

            ax3.set_xlabel("Frame index")
            ax3.set_ylabel("Focus level (µm)")
            ax3.set_title(f"Average Stage Drift (Z) per Frame: {mean_zstep:.4f} µm")
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, "Not enough data for Z drift", ha='center', va='center')
            ax3.axis('off')
    
        plt.tight_layout()

        if __name__ == '__main__':
            plt.show()
    

    valid = mean_step is not None and mean_zstep is not None and not np.isnan(mean_step) and not np.isnan(mean_zstep)
    return {
        "figure": fig if visu else None,
        "figure name": 'stage_stability',
        "status": "success" if valid else "error",
        "measurements": [
            {'category': 'Stage Stability', 'name': 'Mean XY Drift', 'value': mean_step, 'unit': 'µm'},
            {'category': 'Stage Stability', 'name': 'Net XY Drift', 'value': net_drift, 'unit': 'µm'},
            {'category': 'Stage Stability', 'name': 'Mean Z Drift', 'value': mean_zstep, 'unit': 'µm'},
            {'category': 'Stage Stability', 'name': 'Net Z Drift', 'value': net_zdrift, 'unit': 'µm'},
        ],
        "extra_measurements": {},
    }


if __name__ == "__main__": #standalone test
    import pprint
    logger.setup_logger()
    path = r"upload"
    filename = '63x - cross 0.125 time.czi'
    
    img_array, metadata = reader(path+os.sep+filename)
    result = measure_stage_stability(img_array, metadata, debug=True)
    pprint.pprint(result)