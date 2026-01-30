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
from skimage import filters, transform
from skimage.draw import line

# local import
from . import logger
from .functions import (
    centuring_acc,
    crop_with_border,
    dimension_slicer,
    get_centroid,
    img_best_rotation,
    lattice_fitting,
    norm_img,
    sort_grid,
    uniformity,
)

SELECTION_SIZE = 3  # in micron


def get_uniformity(img, grid_size, pxl_size_xy):
    blue = norm_img(img)

    final_angle = img_best_rotation(blue)
    blue = transform.rotate(blue, final_angle, preserve_range=True)
    top, bottom = crop_with_border(blue, pxl_size_xy=pxl_size_xy)
    blue_c = blue[top[0] : bottom[0], top[1] : bottom[1]]
    smoothed_img = filters.gaussian(blue_c, sigma=1.5, preserve_range=True)
    blue_points = get_centroid(smoothed_img, grid_size * grid_size)
    if blue_points is None:
        logger.debug("[Uniformity] get_centroid returned None")
        return None

    lattice = lattice_fitting(
        blue_points,
        grid_size,
        5.0,
        pxl_size_xy,
        return_rawlattice=True,
        image_shape=blue_c.shape,
        detection_percentage=0.98,
    )

    H, W = blue_c.shape
    s = round((SELECTION_SIZE / pxl_size_xy) / 2)
    intensity = []
    points = []

    for coo in lattice:
        x, y = coo[0], coo[1]
        y1, y2 = int(round(y) - s), int(round(y) + s)
        x1, x2 = int(round(x) - s), int(round(x) + s)
        y1c = max(0, y1)
        y2c = min(H, y2)
        x1c = max(0, x1)
        x2c = min(W, x2)

        # Skip if the clipped region has no pixels
        if y2c <= y1c or x2c <= x1c:
            continue

        points.append([x, y])
        area = (y2c - y1c) * (x2c - x1c)
        weight = area / ((2 * s) * (2 * s))  # fraction of expected area
        val = weight * np.nanmean(smoothed_img[y1c:y2c, x1c:x2c])
        intensity.append(val)

    points, sort_indices = sort_grid(np.array(points), gridsize=grid_size)
    intensity = np.array(intensity)[sort_indices]

    if len(intensity) != grid_size * grid_size:
        logger.debug(
            "[Uniformity] Wrong number of intensity points or point outside the image"
        )
        return None

    final_image = intensity.reshape(grid_size, grid_size)
    final_image = filters.median(final_image)
    final_image = final_image * (100 / np.max(final_image))  # to get in percentage

    if np.isnan(np.ravel(final_image)).any():
        logger.debug("[Uniformity] NaN values in final image")
        return None

    field_uniformity = uniformity(final_image)

    return {
        "img": blue_c,
        "uniformity": field_uniformity,
        "points": points,
        "final_image": final_image,
        "sort_indices": sort_indices,
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

    if int(metadata.get("Image Size Z", 0)) < 2:  # single plane
        best_results_uniformity = get_uniformity(
            img_array,
            grid_size,
            pxl_size_xy,
        )
        best_idx = 0
        if best_results_uniformity is None:
            return return_error()
    else:
        results_uniformity = []
        for plane in range(img_array.shape[0]):
            try:
                result = get_uniformity(img_array[plane], grid_size, pxl_size_xy)
            except (ValueError, IndexError, TypeError):
                result = None
            results_uniformity.append(result)
        if debug:
            logger.debug(
                f"[Plane uniformity]: {str([x['uniformity'] if x is not None else 'INF' for x in results_uniformity])}"
            )

        valid_entries = [
            (i, r) for i, r in enumerate(results_uniformity) if r is not None
        ]

        if not valid_entries:
            # total failure
            return return_error()

        best_idx, best_results_uniformity = max(
            valid_entries, key=lambda x: x[1]["uniformity"]
        )

        del results_uniformity

    field_uniformity = best_results_uniformity["uniformity"]
    img = best_results_uniformity["img"]
    points = best_results_uniformity["points"]
    final_image = best_results_uniformity["final_image"]
    sort_indices = best_results_uniformity["sort_indices"]

    # diagonal profiles
    diagonal = np.diag(final_image)
    anti_dia = np.diag(np.fliplr(final_image))
    # min_max profile
    min_coord = np.unravel_index(
        np.argmin(final_image), final_image.shape
    )  # (row, col)
    max_coord = np.unravel_index(np.argmax(final_image), final_image.shape)
    rr, cc = line(min_coord[0], min_coord[1], max_coord[0], max_coord[1])
    line_profile = final_image[rr, cc]

    # stats calculation
    roll_off_dia = np.max(diagonal) - np.min(diagonal)
    roll_off_adi = np.max(anti_dia) - np.min(anti_dia)

    # Compute the geometric center
    geo_center = ((grid_size - 1) / 2, (grid_size - 1) / 2)
    # Compute intensity-weighted centroid
    x_coords, y_coords = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    centroid_x = np.sum(x_coords * final_image) / np.sum(final_image)
    centroid_y = np.sum(y_coords * final_image) / np.sum(final_image)
    centroid = (centroid_x, centroid_y)

    # compute the centering accurancy
    centering_accuracy = centuring_acc(final_image)

    # visualization
    if visu:
        X = points[:, 0][sort_indices].reshape(grid_size, grid_size)
        Y = points[:, 1][sort_indices].reshape(grid_size, grid_size)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        ax1.imshow(img.T, origin="lower")
        ax1.set_title("Reference image")
        ax1.axis("off")

        level = 4
        CS = ax2.contour(
            points[:, 1].reshape(grid_size, grid_size),
            points[:, 0].reshape(grid_size, grid_size),
            filters.gaussian(final_image, 0.8),
            levels=level,
            linewidths=range(level),
            colors="black",
        )
        ax2.clabel(CS, fontsize=10)

        im = ax2.imshow(
            final_image,
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            cmap="jet_r",
            alpha=0.7,
            vmin=0,
            vmax=np.max(final_image),
            origin="lower",
        )

        ax2.plot(
            centroid[0] / final_image.shape[0] * img.shape[0],
            centroid[1] / final_image.shape[1] * img.shape[1],
            "r+",
            markersize=10,
            zorder=100,
        )
        ax2.plot(
            geo_center[0] / final_image.shape[0] * img.shape[0],
            geo_center[1] / final_image.shape[1] * img.shape[1],
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
        ax3.set_ylim([0, np.max(final_image)])
        ax3.set_title("Line intensity profiles")

        fig.subplots_adjust(
            bottom=0.1, left=0.05, right=0.95
        )  # Ensure enough space for both plots and the text

        # Add statistics text at the bottom
        stats_text = (
            f"Roll off Diagonal: {roll_off_dia:.2f}% - "
            f"AntiDiagonal: {roll_off_adi:.2f}%\n"
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
        },
    }
