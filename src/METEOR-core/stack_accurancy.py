# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 10:15:54 2025

@author: simon

Standalone confirmed!

require in metadata:
    'Physical pixel size X'
    'Physical pixel size Z'
    'Image Size Z'
    'Lens Magnification'
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform, measure
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

from functions import (dimension_slicer, reader, norm_center_img_3D,
                       img_best_rotation, rings_detection, aggregate_close_points)
import logger

CORNER_TOL_UM = 2
ROW_BAND_UM = 2

def get_r2(sorted_point:np.array):
    indices = np.arange(len(sorted_point)).reshape(-1, 1)  # Reshape for regression
    
    try:
        # Fit a linear regression model
        model = LinearRegression()
        model.fit(indices, sorted_point)
        predicted = model.predict(indices)
        # Calculate R^2 (linearity)
        r2 = r2_score(sorted_point, predicted)
    
        return predicted, r2
    
    except ValueError: #error during the linear regression
        logger.warning('Error during the linear regression')
        return [], 0


def remove_corners(points_um, corners_um, tol_um=CORNER_TOL_UM):
    """Drop points within tol_um of any corner. points_um: (N,3) in µm (z,y,x)."""
    keep = []
    for p in points_um:
        d = np.linalg.norm(corners_um - p, axis=1)
        if np.min(d) > tol_um:
            keep.append(p)
    return np.array(keep)

def return_error() -> dict:
    return {
        "figure": None,
        "figure name": "Zstack_accuracy",
        "status": "error",
        "measurements": [
            {'category': 'Zstack Accuracy', 'name': 'MAE', 'value': None, 'unit': 'µm'},
            {'category': 'Zstack Accuracy', 'name': 'Precision score', 'value': None, 'unit': ''},
        ],
        "extra_measurements": {},
    }

def stack_precision(img_array:np.ndarray, metadata:dict, visu:bool=True, dim:str='ZYX', debug:bool=False) -> dict:
    """
    Estimate Z-stack accuracy using the Argolight 'stairs' pattern.

    Workflow
    --------
    1. Rotate the stack to align the pattern (angle from middle Z plane).
    2. Normalize / center the 3D image (`norm_center_img_3D`).
    3. Detect all ring objects (`rings_detection` + 3D labeling).
    4. Convert centroids to microns (z, y, x).
    5. Find four corner points (top-left, top-right, bottom-left, bottom-right).
    6. Remove rings close to corners (within `CORNER_TOL_UM`).
    7. Keep only points near the two horizontal stair rows
       (row1 at ~top_y + 14.5 µm, row2 at row1 + 7 µm, within `ROW_BAND_UM`).
    8. Aggregate nearby points into clusters (to merge redundancies).
    9. Extract Z positions for row1 and row2, sort by x, average them to a single
       Z-position sequence, and compute:
         * measured step sizes between adjacent points
         * best matching expected step (1, 0.5, 0.25, 0.125 µm)
         * MAE between measured steps and expected step
         * precision score = 1 - MAE / expected_step

    Parameters
    ----------
    img_array : np.ndarray
        3D stack, shape (Z, Y, X).
    metadata : dict
        Must contain 'Physical pixel size X' and 'Physical pixel size Z' in µm.
    visu : bool
        If True, returns a Matplotlib figure with:
          - detected points overlaid on projection,
          - Z vs point index plot with linear fit.
    debug : bool
        If True, logs extra debug information.

    Returns
    -------
    dict
        {
          "figure": Figure or None,
          "figure name": "Zstack_accuracy",
          "status": "success" or "error",
          "measurements": [
              {"category": "Zstack Accuracy", "name": "MAE", "value": mae, "unit": "µm"},
              {"category": "Zstack Accuracy", "name": "Precision score", "value": precision_score, "unit": ""},
          ],
          "extra_measurements": {...}
        }
    """

    if int(metadata.get('Image Size C', 0)) > 1: #security, in case of multi-C, select the first one
        img_array = img_array[:,:1,...]
        if debug:
            logger.info("stack_precision: multi-C image, taking C=0 only, new shape: "+str(img_array.shape))
    if int(metadata.get('Image Size T', 0)) > 1: #security, in case of multi-T, select the first one
        img_array = img_array[:1,...]
        if debug:
            logger.info("stack_precision: multi-T image, taking T=0 only, new shape: "+str(img_array.shape))
    img_array = dimension_slicer(img_array, metadata, dim)

    img_array = np.asarray(img_array)
    if img_array.ndim != 3:
        raise ValueError(f"stack_precision expects a 3D stack, got shape {img_array.shape}")

    pxl_size_xy = float(metadata.get('Physical pixel size X', 1.0))
    pxl_size_z = float(metadata.get('Physical pixel size Z', 1.0))   
    
    # Prepare, norm and crop the image
    mid = img_array.shape[0]//2
    final_angle = img_best_rotation(img_array[mid])
    rotated_stack = np.stack(
        [transform.rotate(z, final_angle, preserve_range=True) for z in img_array],
        axis=0)
    img_array = norm_center_img_3D(rotated_stack)

    if debug:
        logger.info(f"[stack_precision] img shape after rotation/norm: {img_array.shape}, angle: {final_angle:.2f}°")

    # Detect the rings
    binary = rings_detection(img_array, metadata)
    labeled_img = measure.label(binary)
    regions = measure.regionprops(labeled_img, intensity_image=img_array)

    if not regions:
        logger.warning("[stack_precision] No regions detected.")
        return return_error()

    xyz_list = [x.centroid_weighted for x in regions]
    xyz_arr = np.array(xyz_list)
    xyz_arr_micron = np.copy(xyz_arr)
    xyz_arr_micron[:,0] *= pxl_size_z
    xyz_arr_micron[:,1:] *= pxl_size_xy
    
    #detect the corners
    xy_arr = xyz_arr[:, 1:]
    ymin, xmin = xy_arr.min(axis=0)
    ymax, xmax = xy_arr.max(axis=0)
    corner_targets = np.array([
        [ymin, xmin],  # top-left
        [ymin, xmax],  # top-right
        [ymax, xmin],  # bottom-left
        [ymax, xmax]   # bottom-right
    ])
    
    corner_indices = []
    for ct in corner_targets:
        dists = np.linalg.norm(xy_arr - ct, axis=1)
        corner_indices.append(np.argmin(dists))
    
    corner_points = xyz_arr[corner_indices]                 # in pixel units (z,y,x)
    corner_points_micron = xyz_arr_micron[corner_indices]   # in micron

    #remove the corners
    xyz_arr_micron_filtered = remove_corners(xyz_arr_micron, corner_points_micron, tol_um=CORNER_TOL_UM)

    if xyz_arr_micron_filtered.size == 0:
        logger.warning("[stack_precision] All points removed as corners.")
        return return_error()

    #detect the stairs
    # average of the two top corners
    top_y = (corner_points_micron[0,1] + corner_points_micron[1,1]) / 2       
    row1_y = top_y + 14.5     # first stair row
    row2_y = row1_y + 7.0     # second stair row
    rows_y = [row1_y, row2_y]
    
    # keep only points close to these 2 rows
    xyz_arr_micron_filtered = np.array([
        coo for coo in xyz_arr_micron_filtered
        if any(abs(coo[1] - row_y) < ROW_BAND_UM for row_y in rows_y)
    ])

    if xyz_arr_micron_filtered.size == 0:
        logger.warning("[stack_precision] No points found near stair rows.")
        return return_error()
       
    agg_pts_um, _ = aggregate_close_points(
        xyz_arr_micron_filtered,
        diameter_um=2.0,
        return_groups=False
    )

    if len(agg_pts_um) == 0:
        logger.warning("[stack_precision] No aggregated points after clustering.")
        return return_error()

    # separate row1 / row2 & sort by x
    row1_points = []
    for coo in agg_pts_um:
        if coo[1] > row1_y-ROW_BAND_UM and coo[1] < row1_y+ROW_BAND_UM:
            row1_points.append(coo)
       
    row2_points = []
    for coo in agg_pts_um:
        if coo[1] > row2_y-ROW_BAND_UM and coo[1] < row2_y+ROW_BAND_UM:
            row2_points.append(coo)

    if len(row1_points) == 0 or len(row2_points) == 0:
        logger.warning("[stack_precision] Not enough points in row1 or row2.")
        return return_error()

    row1_points = np.sort(row1_points, axis=0)
    row2_points = np.sort(row2_points, axis=0)

    if len(row1_points) != len(row2_points):
        logger.warning("[stack_precision] Not the same amount of points between row1 and row2.")
        return return_error()

    row1_z = row1_points[:,0]
    row2_z = row2_points[:,0]

    # average Z between rows
    pred_1, r2_1 = get_r2(row1_z)
    pred_2, r2_2 = get_r2(row2_z)
    row_mean_z = np.mean([row1_z, row2_z], axis=0)

    if len(row_mean_z) < 2:
        logger.warning("[stack_precision] Not enough points to compute steps.")
        return return_error()

    measured_steps = np.diff(row_mean_z)
    avg_step = np.mean(measured_steps)
    possible_steps = np.array([1.0, 0.5, 0.25, 0.125])
    closest_step = possible_steps[np.argmin(np.abs(possible_steps - avg_step))]
    
    logger.info(f"Measured average step: {avg_step:.4f} µm")
    logger.info(f"Closest expected step: {closest_step:.3f} µm")

    mae = np.mean(np.abs(measured_steps - closest_step))
    precision_score = 1 - (mae / closest_step) if closest_step > 0 else 0.0
    precision_score = max(0, precision_score)  # clamp to 0..1
    
    fig = None
    if visu:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.imshow(np.sum(img_array, axis=0))
        #all points detected plot
        ax1.scatter(xyz_arr_micron_filtered[:,2]/pxl_size_xy,
                    xyz_arr_micron_filtered[:,1]/pxl_size_xy,
                    c="red", alpha=.5, s=.5,)
        # Correct points
        ax1.scatter(row1_points[:,2]/pxl_size_xy,
                    row1_points[:,1]/pxl_size_xy,
                    c="orange", alpha=.8, s=2, label="Row 1")
        ax1.scatter(row2_points[:,2]/pxl_size_xy,
                    row2_points[:,1]/pxl_size_xy,
                    c="green", alpha=.8, s=2, label="Row 2")
        #corner
        ax1.scatter(corner_points[:,2], corner_points[:,1], c='cyan', s=40, marker='x')
        ax1.axis('off')
        ax1.legend(frameon='off')    
        
        ax2.plot(row_mean_z, label='Average')
        ax2.plot((pred_2+pred_1)/2, '--', label='Prediction')
        ax2.set_ylabel('Z level (μm)')
        ax2.set_xlabel('Sorted points')
        ax2.set_title(f"Z accuracy mae: {mae:.3f}, precision: {precision_score:.3f}")
        ax2.legend(frameon='off') 

        plt.tight_layout()

        if __name__ == "__main__":
            plt.show()

    valid = np.isfinite(mae)
    return {
        "figure": fig if visu else None,
        "figure name": 'Zstack_accuracy',
        "status": "success" if valid else "error",
        "measurements": [
            {'category': 'Zstack Accuracy', 'name': 'MAE', 'value': mae, 'unit': 'µm'},
            {'category': 'Zstack Accuracy', 'name': 'Precision score', 'value': precision_score, 'unit': ''},
        ],
        "extra_measurements": {'r2_1':r2_1,
                               'r2_2':r2_2},
    }

if __name__ == "__main__":
    import os
    import pprint
    logger.setup_logger()
    path = r"upload"
    filename = '20x - cross 1.czi'
    filename = '63x - cross 0.25.czi'
    # filename = '20x - cross 0.5.czi'
    
    img_array, metadata = reader(path+os.sep+filename)
    result = stack_precision(img_array, metadata, visu=True, debug=True)
    pprint.pprint(result)