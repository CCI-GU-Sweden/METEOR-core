"""
src/METEOR_core/intensity_analysis.py

Created on Wed Jun 11 17:00:27 2025

@author: simon

Intensity analysis module for Argolight 4x4 pattern.
"""

# general import
from typing import Any

# analysis import
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from scipy import ndimage, spatial
from skimage import feature, measure
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# local import
from . import logger
from .focus import select_focus_plane
from .functions import (
    dimension_slicer,
    get_label,
    lattice_fitting,
    norm_img,
)


def coverage_of_centers_in_rois(
    centers_xy: np.ndarray, lattice_xy: np.ndarray, half_size_px: float
) -> float:
    """
    centers_xy: (N,2) detected centers (x,y)
    lattice_xy: (M,2) lattice nodes (x,y)
    half_size_px: half ROI size (s) in px (same as you use)
    Returns fraction of centers that land within the ROI of their nearest lattice node.
    """
    tree = spatial.KDTree(lattice_xy)
    _, nn_idx = tree.query(centers_xy)

    assigned = lattice_xy[nn_idx]  # (N,2)
    dx = np.abs(centers_xy[:, 0] - assigned[:, 0])
    dy = np.abs(centers_xy[:, 1] - assigned[:, 1])

    inside = (dx <= half_size_px) & (dy <= half_size_px)
    return float(np.mean(inside)) if len(centers_xy) else 0.0


def angle_deviation_from_right_angles(angle_rad: float) -> float:
    """Return deviation in degrees from nearest multiple of 90°."""
    ang = (angle_rad + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
    # nearest multiple of pi/2
    nearest = np.round(ang / (np.pi / 2)) * (np.pi / 2)
    dev_rad = np.abs(ang - nearest)
    return float(np.degrees(dev_rad))


def get_r2(sorted_intensities: np.ndarray) -> tuple[np.ndarray | None, float | None]:
    """
    Fit a 1D linear regression to sorted intensities and compute R².

    Parameters
    ----------
    sorted_intensities : np.ndarray
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

    indices = np.arange(len(sorted_intensities)).reshape(
        -1, 1
    )  # Reshape for regression
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


def correct_orientation(
    image: np.ndarray,
    labels: np.ndarray,
    expected_corner: str = "top-right",
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
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
        if verbose:
            logger.warning(
                "[correct_orientation] No labeled regions found; skipping orientation correction."
            )
        return image, labels

    largest = max(props, key=lambda r: r.area)
    cy, cx = largest.centroid
    h, w = image.shape

    vertical = "top" if cy < h / 2 else "bottom"
    horizontal = "left" if cx < w / 2 else "right"
    actual_corner = f"{vertical}-{horizontal}"

    if verbose:
        logger.debug(f"Detected: {actual_corner}, expected: {expected_corner}")

    transforms = {
        ("top-left", "top-right"): lambda img: np.fliplr(img),
        ("top-left", "bottom-left"): lambda img: np.flipud(img),
        ("top-left", "bottom-right"): lambda img: np.rot90(img, 2),
        ("top-right", "bottom-left"): lambda img: np.flip(img),
        ("top-right", "top-left"): lambda img: np.fliplr(img),
        ("top-right", "bottom-right"): lambda img: np.rot90(img, 3),
        # Warning, not complete! 'bottom-left' and 'bottom-right' not used! #TODO
    }

    key = (expected_corner, actual_corner)
    if expected_corner != actual_corner and key in transforms:
        transform_fn = transforms[key]
        image = transform_fn(image)
        labels = transform_fn(labels)
        if verbose:
            logger.debug(f"Image flipped to align {actual_corner} → {expected_corner}")
    elif expected_corner != actual_corner and key not in transforms and verbose:
        logger.debug(
            f"No transform defined for (expected={expected_corner}, actual={actual_corner}); leaving as-is."
        )

    return image, labels


def get_intensity(
    img: np.ndarray,
    metadata: dict,
    grid_size: int = 4,
    pattern_size: float = 8.0,
    spacing: float = 9.5,
    threshold_method: str = "otsu",
    closing_radius: int = 2,
    visu=False,
    debug=False,
) -> tuple[float | None, Figure | None, dict[str, Any] | None]:
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
    fig : Figure or None
        Figure object if `visu=True`, else None.
    info : dict or None
        Additional information dictionary if `visu=True`, else None.
    """
    pxl_size_xy = float(metadata.get("Physical pixel size X", 1.0))

    img = norm_img(img)
    labels = get_label(
        img, thr=threshold_method, closing_radius=closing_radius, debug=debug
    )

    if debug:
        logger.debug(f"[intensity_analysis] Number of labels detected: {labels.max()}")
    if np.max(labels) == 0:
        if debug:
            logger.warning("[intensity_analysis] No regions detected in label image.")
        return None, None, None

    # Orientation correction: largest ROI should be in top-right by default
    img, labels = correct_orientation(img, labels, verbose=debug)

    # peak detection on distance transform
    dist = ndimage.distance_transform_edt(labels > 0)
    half_size_px = max(1, int(round((pattern_size / pxl_size_xy) / 2)))
    min_distance = max(1, int(round(0.6 * half_size_px)))

    peaks_rc = feature.peak_local_max(
        dist,
        min_distance=min_distance,
        labels=labels > 0,
        exclude_border=True,
    )
    centers = np.column_stack([peaks_rc[:, 1], peaks_rc[:, 0]]).astype(float)

    try:
        lattice = lattice_fitting(
            centers,
            grid_size,
            spacing,
            pxl_size_xy,
            return_rawlattice=True,
            return_params=True,
            image_shape=img.shape,
        )
    except (ValueError, RuntimeError) as e:
        if debug:
            logger.warning(f"[intensity_analysis] Lattice_fitting failed: {e}")
        return None, None, None

    # grab a selection around the estimated center and measure intensity in it
    s = round((pattern_size / pxl_size_xy) / 2)
    intensities = []
    rect_list = []
    H, W = img.shape

    for coo in lattice[0]:
        x, y = coo[0], coo[1]
        y1, y2 = int(round(y)) - s, int(round(y)) + s
        x1, x2 = int(round(x)) - s, int(round(x)) + s

        if (y1 < 0) or (x1 < 0) or (y2 > H) or (x2 > W):
            if debug:
                logger.warning(
                    "[intensity_analysis] ROI out of bounds -> fitting likely out of FOV."
                )
            return None, None, None

        if (y2 <= y1) or (x2 <= x1):
            intensities.append(np.nan)
        else:
            intensities.append(np.nanmean(img[y1:y2, x1:x2]))
        # save the rectangle
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red")
        rect_list.append(rect)

    intensities = np.asarray(intensities, dtype=float)

    # background value estimation (simple)
    bg_pixels = img[labels == 0]
    bg = np.mean(bg_pixels)
    bg_sigma = np.median(np.abs(bg_pixels - bg))  # robust std via MAD

    intensities = intensities - bg
    intensities[intensities < bg_sigma] = 0.0  # zero-out low values
    if intensities.size - np.count_nonzero(intensities) >= 3:
        if debug:
            logger.warning(
                "[intensity_analysis] Too many low intensities after background subtraction."
            )
        return None, None, None

    # robust dynamic range of ROI means
    p10 = np.nanpercentile(intensities, 10)
    p90 = np.nanpercentile(intensities, 90)
    dyn_range = (p90 - p10) / (p90 + 1e-12)  # 0..~1

    # how "non-flat" the pattern is (another simple measure)
    cv = float(np.nanstd(intensities) / (np.nanmean(intensities) + 1e-12))
    dev_deg = angle_deviation_from_right_angles(lattice[1][0][2])
    coverage = coverage_of_centers_in_rois(centers, lattice[0], half_size_px=s)

    # gates
    if coverage < 0.75:
        score = None
    else:
        # normalize angle penalty (0 penalty at 0°, strong after 10°)
        ang_pen = max(0.0, (dev_deg - 5.0) / 35.0)
        ang_pen = min(1.0, ang_pen)

        # prefer lower fit cost (use log so huge costs don't dominate)
        cost_term = -np.log10(lattice[1][1] + 1e-9)

        # combine
        score = 2.0 * dyn_range + 1.0 * cv + 0.25 * cost_term - 1.0 * ang_pen

    if debug:
        logger.debug(
            f"[intensity_analysis] Quality metrics: dyn_range={dyn_range:.3f}, cv={cv:.3f}, "
            f"angle_dev={dev_deg:.2f}°, coverage={coverage:.3f}, fit_cost={lattice[1][1]:.3e}, score={score}"
        )
    if score is None or score < 0.5:
        if debug:
            logger.warning(
                f"[intensity_analysis] Quality checks failed (score={score}); cannot compute reliable R²."
            )
        return None, None, None

    sorted_intensities = np.sort(intensities)
    r2 = None
    predicted = None
    if np.isnan(sorted_intensities).any():
        if debug:
            logger.warning(
                "[intensity_analysis] NaNs found in intensities; cannot compute a reliable R²."
            )
    else:
        predicted, r2 = get_r2(sorted_intensities)
        if r2 is None:
            if debug:
                logger.warning("[intensity_analysis] Failed to compute R².")
            predicted = None

    # visu
    fig = None
    if visu:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        # image and pattern fit
        from skimage import color, morphology

        edge_labels = (
            morphology.dilation(labels, morphology.disk(3)) - labels
        )  # edge only
        edge_labels = color.label2rgb(edge_labels)
        ax1.imshow(edge_labels)
        img_to_display = img * (labels > 0)
        img_to_display[img_to_display == 0] = np.nan
        ax1.imshow(img_to_display, cmap="gray")  # only show labeled area
        ax1.scatter(centers[:, 0], centers[:, 1], c="r", s=10, label="Detected")
        ax1.scatter(lattice[0][:, 0], lattice[0][:, 1], c="g", s=5, label="Selected")  # type: ignore
        for rect in rect_list:
            ax1.add_patch(rect)
        ax1.axis("off")
        ax1.set_title("Detected 4x4 pattern")
        ax1.legend()
        # Plot the sorted intensities and linear fit
        ax2.plot(sorted_intensities, label="Sorted Intensities", marker="o")
        ax2.plot(
            predicted, label=f"Linear Fit (R²={r2:.4f})", linestyle="--", color="red"
        )
        ax2.set_xlabel("Index (Sorted)")
        ax2.set_ylabel("Normalized Intensity (a.u.)")
        ax2.set_title("Linearity of Intensities")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        if __name__ == "__main__":
            plt.show()

    info = {
        "score": score,
        "valid": bool,
        "coverage": coverage,
        "angle_dev_deg": dev_deg,
        "thr": threshold_method,
    }

    return r2, fig, info


def intensity_analysis(
    img_array: np.ndarray,
    metadata: dict,
    grid_size: int = 4,
    pattern_size: float = 8.0,
    spacing: float = 10.0,
    dim="ZYX",
    debug: bool = False,
    stop_value: float = 0.995,
) -> dict[str, Any]:
    """
    High-level intensity linearity analysis.

    - If the image is 2D or metadata says single Z, compute on that plane.
    - Select the Z plane using an image-focus score independent of linearity.

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
    stop_value : float
        Retained for API compatibility; no longer used for plane selection.

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
    if (
        int(metadata.get("Image Size C", 0)) > 1
    ):  # security, in case of multi-C, select the first one
        img_array = img_array[:, :1, ...]
        if debug:
            logger.debug(
                f"[intensity analysis]: taking first channel for analysis: shape now {img_array.shape}"
            )
    if (
        int(metadata.get("Image Size T", 0)) > 1
    ):  # in case of multi time points, take the first one
        img_array = img_array[:1, ...]
        if debug:
            logger.debug(
                f"[intensity analysis]: taking first time point for analysis: shape now {img_array.shape}"
            )
    img_array = dimension_slicer(img_array, metadata, dim)
    # check that the image as at least one Z plane
    if img_array.ndim == 2:
        img_array = img_array[np.newaxis, ...]  # shape ZXY

    best = {"r2": None, "plane": None, "thr": None}

    THRESH_METHODS = ["li", "yen", "triangle", "otsu", "mean"]

    candidates = []

    try:
        plane, focus_scores = select_focus_plane(img_array)
    except ValueError as exc:
        logger.warning(f"Intensity focus selection failed: {exc}")
        plane, focus_scores = None, []
    if plane is not None:
        for thr in THRESH_METHODS:
            _, _, info = get_intensity(
                img_array[plane],
                metadata,
                grid_size=grid_size,
                pattern_size=pattern_size,
                spacing=spacing,
                threshold_method=thr,
                visu=False,
                debug=debug,
            )
            if info is not None:
                info = dict(info)
                info["plane"] = plane
                info["thr"] = thr
                candidates.append(info)

    if not candidates:  # no result at all! Error
        return {
            "figure": None,
            "figure name": "Intensity_r2",
            "status": "error",
            "measurements": [
                {
                    "category": "Intensity",
                    "name": "Linearity of Intensity",
                    "value": None,
                    "unit": "",
                }
            ],
        }

    best = max(
        candidates, key=lambda c: c.get("score") if c.get("score") is not None else -1e9
    )

    r2, fig, _ = get_intensity(
        img_array[best["plane"]],
        metadata,
        pattern_size=pattern_size,
        spacing=spacing,
        grid_size=grid_size,
        threshold_method=str(best["thr"]),
        visu=True,
        debug=debug,
    )

    valid = r2 is not None and r2 > 0
    return {
        "figure": fig,
        "figure name": "Intensity_r2",
        "status": "success" if valid else "error",
        "measurements": [
            {
                "category": "Intensity",
                "name": "Linearity of Intensity",
                "value": r2,
                "unit": "",
            }
        ],
        "extra_measurements": {
            "best_z_plane": best["plane"],
            "used_threshold": str(best["thr"]),
            "fitting_score": best.get("score"),
            "focus_scores": focus_scores,
            "focus_at_stack_boundary": len(focus_scores) > 1
            and best["plane"] in (0, len(focus_scores) - 1),
        },
    }
