# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 17:00:27 2025

@author: simon

Simple program to analyze the intensity of the Argolight 4x4 intensity pattern.
1st: detecting the 4x4 grid
2nd: extract intensity and sort them
3rd: linar interpolation -> r square

Small trick: in case of a 3D image, it will process all 2D planes, then select the highest r square as correct value

"""

#general import
from typing import Dict, Any, List, Tuple
import numpy as np
import os

#analysis import
import matplotlib.pyplot as plt
from skimage import measure
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#local import
from . import logger
from .functions import dimension_slicer, norm_img, get_label, reader, lattice_fitting


def get_r2(sorted_intensities:List[float]) -> Tuple[np.ndarray, float]:
    """
    Fit a 1D linear regression to sorted intensities and compute R².

    Parameters
    ----------
    sorted_intensities : list of float
        Intensities sorted in ascending order (or any order; index is used as X).

    Returns
    -------
    predicted : np.ndarray or None
        Predicted intensities from the linear model, same length as input.
        Returns None if regression cannot be performed.
    r2 : float or None
        Coefficient of determination (R²). None if regression fails.
    """
    n = len(sorted_intensities)
    if n == 0:
        logger.warning("[get_r2] empty intensity list; cannot compute R².")
        return None, None

    indices = np.arange(len(sorted_intensities)).reshape(-1, 1)  # Reshape for regression
    y = np.asarray(sorted_intensities, dtype=float)

    try:
        model = LinearRegression()
        model.fit(indices, y)
        predicted = model.predict(indices)
        r2 = r2_score(y, predicted)
        return predicted, r2
    except ValueError as e:
        logger.warning(f"Error during linear regression in get_r2: {e}")
        return None, None


def correct_orientation(image:np.ndarray, labels:np.ndarray,
                        expected_corner:str='top-right',
                        verbose:bool=False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Automatically flip/rotate image and labels so that the largest ROI is in the expected corner.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale image (Y, X).
    labels : np.ndarray
        2D labeled image (same shape as `image`).
    expected_corner : {'top-left', 'top-right', 'bottom-left', 'bottom-right'}
        Desired location of the largest ROI. Default is 'top-right'
    verbose : bool
        If True, log detailed orientation information.

    Returns
    -------
    image_corr : np.ndarray
        Possibly flipped/rotated image.
    labels_corr : np.ndarray
        Transformed labels (same transformation as `image`).
    """

    props = measure.regionprops(labels, intensity_image=image)

    if not props:
        logger.warning("[correct_orientation] no labeled regions found; skipping orientation correction.")
        return image, labels

    largest = max(props, key=lambda r: r.area)
    cy, cx = largest.centroid
    h, w = image.shape

    vertical = 'top' if cy < h / 2 else 'bottom'
    horizontal = 'left' if cx < w / 2 else 'right'
    actual_corner = f"{vertical}-{horizontal}"

    if verbose:
        logger.debug(f"Detected: {actual_corner}, expected: {expected_corner}")

    transforms = {
        ('top-left', 'top-right'): lambda img: np.fliplr(img),
        ('top-left', 'bottom-left'): lambda img: np.flipud(img),
        ('top-left', 'bottom-right'): lambda img: np.rot90(img, 2),
        ('top-right', 'bottom-left'): lambda img: np.flip(img),
        ('top-right', 'top-left'): lambda img: np.fliplr(img),
        ('top-right', 'bottom-right'): lambda img: np.rot90(img, 3),
        #Warning, not complete! 'bottom-left' and 'bottom-right' not used! #TODO
    }

    key = (expected_corner, actual_corner)
    if expected_corner != actual_corner and key in transforms:
        transform_fn = transforms[key]
        image = transform_fn(image)
        labels = transform_fn(labels)
        if verbose:
            logger.debug(f"Image flipped to align {actual_corner} → {expected_corner}")
    elif expected_corner != actual_corner and key not in transforms and verbose:
        logger.debug(f"No transform defined for (expected={expected_corner}, actual={actual_corner}); leaving as-is.")

    return image, labels


def get_intensity(img:np.ndarray, metadata:dict, grid_size:int=4,
                  pattern_size:float=8., spacing:float=10., visu=False,
                  debug=False) -> Tuple[float, plt.Figure]:
    """
    Compute the intensity linearity (R²) of the Argolight 4×4 intensity pattern
    for a single 2D image.

    Steps:
    1. Normalize the image and label the 4×4 pattern.
    2. Fit an ideal 4×4 lattice in physical units (spacing in µm).
    3. Extract mean intensity in small ROIs around each lattice point.
    4. Sort intensities and fit a linear model vs index.
    5. Return R² and an optional figure.

    Parameters
    ----------
    img : np.ndarray
        2D image (Y, X) or single-plane extracted from a stack.
    metadata : dict
        Must contain 'Physical pixel size X' (µm).
    grid_size : int, optional
        Size of the grid along one axis (e.g. 4 for a 4×4 pattern).
    pattern_size : float, optional
        Size of each pattern element in µm (side length of square ROI).
    spacing : float, optional
        Physical spacing between pattern elements (µm).
    visu : bool, optional
        If True, returns a figure with:
            - detected vs fitted centers and ROIs,
            - sorted intensities and linear fit.

    Returns
    -------
    r2 : float or None
        R² of the linear fit of sorted intensities. None if computation fails.
    fig : matplotlib.figure.Figure or None
        Figure object if `visu=True`, else None.
    """
    pxl_size_xy = float(metadata.get('Physical pixel size X', 1.0))

    img = norm_img(img)
    labels = get_label(img)

    # Orientation correction: largest ROI should be in top-right by default
    img, labels = correct_orientation(img, labels, verbose=debug)

    #grab the center
    centers = []
    for region in measure.regionprops(labels):
        y, x = region.centroid
        centers.append((x, y))
    centers = np.array(centers)

    if centers.size == 0:
        logger.warning("get_intensity: no regions detected in label image.")
        return None, None

    try:
        lattice = lattice_fitting(centers, grid_size, spacing, pxl_size_xy,
                                  return_rawlattice=True, image_shape=img.shape)
    except Exception as e:
        logger.warning(f"get_intensity: lattice_fitting failed: {e}")
        return None, None
    
    #grab a selection around the estimated center and measure intensity in it
    s = round((pattern_size/pxl_size_xy)/2)
    intensities = []
    rect_list = []
    H, W = img.shape
    
    for coo in lattice:
        x, y = coo[0], coo[1]
        y1, y2 = int(round(y)) - s, int(round(y)) + s
        x1, x2 = int(round(x)) - s, int(round(x)) + s

        # clip to bounds
        y1 = max(0, y1)
        y2 = min(H, y2)
        x1 = max(0, x1)
        x2 = min(W, x2)

        if (y2 <= y1) or (x2 <= x1):
            intensities.append(np.nan)
        else:
            intensities.append(np.nanmean(img[y1:y2, x1:x2]))
        #save the rectangle
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red')
        rect_list.append(rect)    

    intensities = np.asarray(intensities, dtype=float)
    sorted_intensities = np.sort(intensities)
    if debug:
        logger.debug(f"Length of intensities: {len(intensities)}")

    if np.isnan(sorted_intensities).any():
        logger.warning("get_intensity: NaNs found in intensities; cannot compute a reliable R².")
        r2 = None
        predicted = None
    else:
        predicted, r2 = get_r2(sorted_intensities)
        if r2 is None:
            logger.warning("get_intensity: failed to compute R².")
            predicted = None
    
    #visu
    fig = None
    if visu:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        #image and pattern fit
        ax1.imshow(img, cmap='gray')
        ax1.plot(centers[:,0], centers[:,1], 'r.', label='Detected')
        ax1.plot(lattice[:,0], lattice[:,1], 'g.', label='Selected')
        for rect in rect_list:
            ax1.add_patch(rect)
        ax1.axis('off')
        ax1.set_title("Detected 4x4 pattern")
        ax1.legend()
        # Plot the sorted intensities and linear fit
        ax2.plot(sorted_intensities, label="Sorted Intensities", marker='o')
        ax2.plot(predicted, label=f"Linear Fit (R²={r2:.4f})", linestyle='--', color='red')
        ax2.set_xlabel("Index (Sorted)")
        ax2.set_ylabel("Normalized Intensity (a.u.)")
        ax2.set_title("Linearity of Intensities")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        if __name__ == '__main__':
            plt.show()
    
    return r2, fig    

def intensity_analysis(img_array:np.ndarray, metadata:dict,
                       grid_size:int=4, pattern_size:float=8.,
                       spacing:float=10., dim="ZYX",
                       debug:bool=False,) -> Dict[str, Any]:
    """
    High-level intensity linearity analysis.

    - If the image is 2D or metadata says single Z, compute on that plane.
    - If the image is a Z-stack, compute linearity for each Z plane independently
      and keep the plane with the highest R².

    Parameters
    ----------
    img_array : np.ndarray
        2D image (Y, X) or 3D stack (Z, Y, X).
    metadata : dict
        Must contain 'Image Size Z' and 'Physical pixel size X'.
    grid_size : int
        4 for the Argolight 4×4 pattern.
    pattern_size : float
        Physical size of each pattern element (µm).
    spacing : float
        Physical spacing between elements (µm).

    Returns
    -------
    dict
        {
            "figure": Figure or None,
            "figure name": "Intensity_r2",
            "status": "success" or "error",
            "measurements": [
                {
                    "category": "Intensity",
                    "name": "Linearity of Intensity",
                    "value": r2,
                    "unit": "",
                }
            ],
        }
    """
    image_size_z = int(metadata.get('Image Size Z', 0))
    if int(metadata.get('Image Size C', 0)) > 1: #security, in case of multi-C, select the first one
        img_array = img_array[:,:1,...]
        if debug:
            logger.debug(f"[intensity analysis]: taking first channel for analysis: shape now {img_array.shape}")
    if int(metadata.get('Image Size T', 0)) > 1: # in case of multi time points, take the first one
        img_array = img_array[:1,...]
        if debug:
            logger.debug(f"[intensity analysis]: taking first time point for analysis: shape now {img_array.shape}")
    img_array = dimension_slicer(img_array, metadata, dim) 

    #in case a single Z, can still process the data!
    if (image_size_z < 2 and img_array.ndim == 2) or img_array.ndim == 2:
        r2, fig = get_intensity(img_array, metadata, pattern_size=pattern_size,
                                    spacing=spacing, grid_size=grid_size, visu=True, debug=debug)
        
    else: #assume multi-Z
        results_linearity: List[float | None] = []
        for plane in range(img_array.shape[0]): #shape ZXY!
            try:
                linearity, _ = (get_intensity(img_array[plane], metadata,
                                                grid_size=grid_size,
                                                pattern_size=pattern_size,
                                                spacing=spacing, visu=False))
            except Exception as e:
                logger.debug(f"[intensity_analysis] get_intensity failed at Z={plane}: {e}")
                linearity = None

            results_linearity.append(linearity)

        valid_indices = [i for i, r in enumerate(results_linearity) if r is not None]

        if not valid_indices:
            logger.warning("Intensity linearity analysis failed for all planes (all R² None).")
            return {
                "figure": None,
                "figure name": 'Intensity_r2',
                "status": "error",
                "measurements": [{'category': 'Intensity', 'name': 'Linearity of Intensity', 'value': None, 'unit': '',}]
                }

        best_plane = max(valid_indices, key=lambda i: results_linearity[i])
        r2, fig = get_intensity(img_array[best_plane], metadata, pattern_size=pattern_size,
                                    spacing=spacing, grid_size=grid_size, visu=True, debug=debug)

    valid = r2 is not None and r2 > 0
    return {
        "figure": fig,
        "figure name": 'Intensity_r2',
        "status": "success" if valid else "error",
        "measurements": [{'category': 'Intensity', 'name': 'Linearity of Intensity', 'value': r2, 'unit': '',}],
        "extra_measurements":{'best_z_plane':best_plane}}

if __name__ == '__main__':
    logger.setup_logger()
    import pprint
    path = r"upload"
    filename = 'Argo 20x intensity.czi'
    filename = 'Argo 63x Oil intensity.czi'
    img_array, metadata = reader(path+os.sep+filename)
    result = intensity_analysis(img_array, metadata, debug=True,)
    pprint.pprint(result)