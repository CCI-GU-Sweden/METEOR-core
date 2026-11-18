"""
src/METEOR_core/uniformity_analysis.py

Created on Fri Jun 13 13:53:16 2025

@author: simon

https://pmc.ncbi.nlm.nih.gov/articles/PMC9526251/

Uniformity = 100*Intensity_min/Intensity_max
Centering = 100-(100*((Xmax-Xcenter)**2+(Ymax-Ycenter)**2)**0.5*(2/(x**2+y**2)**0.5))
Xmax and Ymax are the coordinates of the maximum intensity zone
Xcenter and Ycenter are the coordinates of the center of the FOV
x and y determine the width and the height of the image
"""

# general import
from typing import Any

# analysis import
import matplotlib.pyplot as plt
import numpy as np
from skimage import filters
from skimage.draw import line

# local import
from . import logger
from .focus import select_focus_plane
from .functions import (
    centuring_acc,
    dimension_slicer,
    norm_img,
)
from .ring_lattice import detect_ring_lattice

SELECTION_SIZE = 3  # in micron


def get_uniformity(img, grid_size, pxl_size_xy):
    blue = norm_img(img)

    blue_c = blue
    smoothed_img = filters.gaussian(blue_c, sigma=1.5, preserve_range=True)
    registration = detect_ring_lattice(smoothed_img, 5.0 / pxl_size_xy, grid_size)
    if registration.diagnostics["index_extent"] != [grid_size, grid_size]:
        logger.warning("Uniformity requires a grid spanning the full expected field")
        return None

    H, W = blue_c.shape
    s = max(1, round((SELECTION_SIZE / pxl_size_xy) / 2))
    final_image = np.full((grid_size, grid_size), np.nan)
    indices = registration.indices - registration.indices.min(axis=0)
    clipped = 0
    for coo, (ix, iy) in zip(registration.points_xy, indices, strict=False):
        x, y = coo[0], coo[1]
        y1, y2 = int(round(y) - s), int(round(y) + s)
        x1, x2 = int(round(x) - s), int(round(x) + s)
        if y1 < 0 or x1 < 0 or y2 > H or x2 > W:
            clipped += 1
            continue
        final_image[iy, ix] = np.mean(smoothed_img[y1:y2, x1:x2])
    count = int(np.isfinite(final_image).sum())
    if count < 0.7 * grid_size * grid_size or np.nanmax(final_image) <= 0:
        logger.warning("Insufficient valid ring ROIs for field uniformity")
        return None
    final_image *= 100 / np.nanmax(final_image)
    field_uniformity = 100 * np.nanmin(final_image) / np.nanmax(final_image)

    return {
        "img": blue_c,
        "uniformity": field_uniformity,
        "points": registration.points_xy,
        "final_image": final_image,
        "registration": registration.diagnostics,
        "valid_roi_count": count,
        "clipped_roi_count": clipped,
    }


def return_error() -> dict[str, Any]:
    return {
        "figure": None,
        "figure name": "Field_Uniformity",
        "status": "error",
        "measurements": [
            {
                "category": "Uniformity - Argolight",
                "name": "Uniformity",
                "value": None,
                "unit": "",
            }
        ],
    }


def uniformity_analysis(
    img_array: np.ndarray,
    metadata: dict,
    grid_size: int = 21,
    visu: bool = True,
    debug: bool = False,
    dim="ZYX",
) -> dict[str, Any]:
    """
    Compute the field uniformity via a 2d rings target.
    Parameters
    ----------
    img_array : np.ndarray
        Image array with multiple Z planes, shape ZYX.
    grid_size : int
        Number of points in the grid (e.g., 21 for a 21x21 grid).
    metadata : Dict[str, Any]
        Metadata dictionary containing at least 'Physical pixel size X' (µm), "axis_order",
        'Image Size C' (number of channels) and 'Image Size Z' (number of Z planes)
    visu : bool, optional
        Whether to generate a visualization figure. The default is True.
    debug : bool, optional
        Whether to enable debug logging. The default is False.
    Returns
    -------
    Dict[str, Any]
        A dictionary containing the analysis results, including the figure (if visu=True),
        status, measurements, and extra measurements.
    """

    pxl_size_xy = float(metadata.get("Physical pixel size X", 1.0))
    if (
        int(metadata.get("Image Size C", 0)) > 1
    ):  # in case of multi channel, take the first one
        img_array = img_array[:, :1, ...]
        if debug:
            logger.debug(
                f"[Plane uniformity]: taking first channel for analysis: shape now {img_array.shape}"
            )
    if (
        int(metadata.get("Image Size T", 0)) > 1
    ):  # in case of multi time points, take the first one
        img_array = img_array[:1, ...]
        if debug:
            logger.debug(
                f"[Plane uniformity]: taking first time point for analysis: shape now {img_array.shape}"
            )
    img_array = dimension_slicer(img_array, metadata, dim)

    try:
        best_idx, focus_scores = select_focus_plane(img_array)
        best_results_uniformity = get_uniformity(
            img_array[best_idx], grid_size, pxl_size_xy
        )
    except (ValueError, IndexError, TypeError) as exc:
        logger.warning(f"Uniformity focus selection or analysis failed: {exc}")
        return return_error()
    if best_results_uniformity is None:
        return return_error()

    field_uniformity = best_results_uniformity["uniformity"]
    img = best_results_uniformity["img"]
    points = best_results_uniformity["points"]
    final_image = best_results_uniformity["final_image"]

    # diagonal profiles
    diagonal = np.diag(final_image)
    anti_dia = np.diag(np.fliplr(final_image))
    # min_max profile
    min_coord = np.unravel_index(
        np.nanargmin(final_image), final_image.shape
    )  # (row, col)
    max_coord = np.unravel_index(np.nanargmax(final_image), final_image.shape)
    rr, cc = line(min_coord[0], min_coord[1], max_coord[0], max_coord[1])
    line_profile = final_image[rr, cc]

    # stats calculation
    roll_off_dia = (
        np.nanmax(diagonal) - np.nanmin(diagonal)
        if np.isfinite(diagonal).any()
        else None
    )
    roll_off_adi = (
        np.nanmax(anti_dia) - np.nanmin(anti_dia)
        if np.isfinite(anti_dia).any()
        else None
    )

    # Compute the geometric center
    geo_center = ((grid_size - 1) / 2, (grid_size - 1) / 2)
    # Compute intensity-weighted centroid
    x_coords, y_coords = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    centroid_x = np.nansum(x_coords * final_image) / np.nansum(final_image)
    centroid_y = np.nansum(y_coords * final_image) / np.nansum(final_image)
    centroid = (centroid_x, centroid_y)

    # compute the centering accurancy
    centering_accuracy = centuring_acc(np.nan_to_num(final_image, nan=-np.inf))

    # visualization
    if visu:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        ax1.imshow(img, cmap="gray")
        ax1.scatter(points[:, 0], points[:, 1], s=3, c="red")
        ax1.set_title("Reference image")
        ax1.axis("off")

        im = ax2.imshow(
            final_image,
            cmap="jet_r",
            vmin=0,
            vmax=np.nanmax(final_image),
        )

        ax2.plot(
            centroid[0],
            centroid[1],
            "r+",
            markersize=10,
            zorder=100,
        )
        ax2.plot(
            geo_center[0],
            geo_center[1],
            "k+",
            markersize=10,
        )

        ax2.set_title("Field Uniformity")
        ax2.axis("off")
        plt.colorbar(im, label="Intensity (%)")

        ax3.plot(diagonal, "ro", linestyle="--", label="Main diagonal")
        ax3.plot(anti_dia, "go", linestyle="--", label="Anti diagonal")
        ax3.plot(line_profile, "ko", linestyle="--", label="Min-Max profile")
        ax3.set_xlabel("Step (/5 microns)")
        ax3.legend()
        ax3.set_ylim([0, np.nanmax(final_image)])
        ax3.set_title("Line intensity profiles")

        fig.subplots_adjust(
            bottom=0.1, left=0.05, right=0.95
        )  # Ensure enough space for both plots and the text

        # Add statistics text at the bottom
        stats_text = (
            f"Roll off Diagonal: {roll_off_dia if roll_off_dia is not None else 'n/a'} - "
            f"AntiDiagonal: {roll_off_adi if roll_off_adi is not None else 'n/a'}\n"
            f"Centering Accuracy: {centering_accuracy:.2f}% - "
            f"Field Uniformity: {field_uniformity:.2f}%"
        )

        # Adding the statistics as text
        fig.text(
            0.5,
            0.08,
            stats_text,
            ha="center",
            va="top",
            fontsize=10,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.5},
        )

        if __name__ == "__main__":
            plt.show()
    else:
        fig = None

    valid = (centering_accuracy is not None) and np.isfinite(field_uniformity)
    return {
        "figure": fig if visu else None,
        "figure name": "Field_Uniformity",
        "status": "success" if valid else "error",
        "measurements": [
            {
                "category": "Uniformity - Argolight",
                "name": "Centering accuracy",
                "value": centering_accuracy,
                "unit": "",
            },
            {
                "category": "Uniformity - Argolight",
                "name": "Uniformity",
                "value": field_uniformity,
                "unit": "",
            },
        ],
        "extra_measurements": {
            "roll_off_dia": roll_off_dia,
            "roll_off_adi": roll_off_adi,
            "best_z_plane": best_idx,
            "focus_scores": focus_scores,
            "focus_at_stack_boundary": len(focus_scores) > 1
            and best_idx in (0, len(focus_scores) - 1),
            "ring_registration": best_results_uniformity["registration"],
            "valid_roi_count": best_results_uniformity["valid_roi_count"],
            "clipped_roi_count": best_results_uniformity["clipped_roi_count"],
        },
    }
