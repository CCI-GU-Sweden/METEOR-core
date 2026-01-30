"""
src/METEOR_core/distortion_analysis.py
Created on Fri Jun 13 14:52:57 2025

@author: simon

metadata needed:
- 'Physical pixel size X' (µm)
- 'Image Size Z'
- 'Image Size C'

"""

# general import
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from skimage import filters, transform

# local import
from . import logger
from .functions import (
    crop_with_border,
    dimension_slicer,
    get_centroid,
    img_best_rotation,
    lattice_fitting,
    norm_img,
    sort_grid,
)


def get_distortion(
    img: np.ndarray,
    pxl_size_xy: float,
    spacing: float = 4.9,
    grid_size: int = 21,
) -> dict[str, Any] | None:
    # normalize the image first
    img = norm_img(img)

    # rotate the image
    final_angle = img_best_rotation(img)
    img = transform.rotate(img, final_angle, preserve_range=True)
    # crop the image
    top_left, bottom_right = crop_with_border(img, pxl_size_xy=pxl_size_xy)
    img = img[top_left[0] : bottom_right[0], top_left[1] : bottom_right[1]]

    points = get_centroid(
        img, grid_size * grid_size, min_size=int(spacing / pxl_size_xy * 0.6)
    )
    if points is None:
        logger.warning(
            "[Distortion analysis]: Number of point detected do not match expected grid size."
        )
        return None

    paired_lattice, sorted_points = lattice_fitting(
        points,
        grid_size,
        spacing,
        pxl_size_xy,
        return_points=True,
        image_shape=img.shape,
    )
    def_vector = paired_lattice - sorted_points
    amplitudes = np.linalg.norm(def_vector, axis=1)

    return {
        "paired_lattice": paired_lattice,
        "points": sorted_points,
        "max_amplitude": np.max(amplitudes),
        "img": img,
    }


def distortion_analysis(
    img_array: np.ndarray,
    metadata: dict,
    grid_size: int = 21,
    spacing: float = 5.1,
    visu: bool = True,
    debug: bool = False,
    dim="ZYX",
) -> dict[str, Any]:
    """
    Compute the field distortion via a 2d rings target.
    Parameters
    ----------
    img_array : np.ndarray
        Image array with multiple Z planes, shape ZYX.
    grid_size : int
        Number of points in the grid (e.g., 21 for a 21x21 grid).
    spacing : float
        Spacing between points in microns.
    metadata : Dict[str, Any]
        Metadata dictionary containing at least 'Physical pixel size X' (µm)
         and 'Image Size Z' (number of Z planes)
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
        int(metadata.get("Image Size T", 0)) > 1
    ):  # security, in case of multi-T, select the first one
        img_array = img_array[:1, ...]
        if debug:
            logger.info(
                "Plane uniformity: multi-T image, taking T=0 only, new shape: "
                + str(img_array.shape)
            )

    if (
        int(metadata.get("Image Size C", 0)) > 1
    ):  # security, in case of multi-C, select the first one
        img_array = img_array[:, :1, ...]
        if debug:
            logger.info(
                "Plane uniformity: multi-C image, taking C=0 only, new shape: "
                + str(img_array.shape)
            )

    img_array = dimension_slicer(img_array, metadata, dim)

    if int(metadata.get("Image Size Z", 0)) < 2:  # single plane
        best_result_distortion = get_distortion(
            img_array, pxl_size_xy, spacing=spacing, grid_size=grid_size
        )
        best_idx = 0
        if best_result_distortion is None:
            return {
                "figure": None,
                "figure name": "Distortion",
                "status": "error",
                "measurements": [
                    {
                        "category": "Distortion",
                        "name": "Amplitude",
                        "value": None,
                        "unit": "µm",
                    }
                ],
            }

    else:
        results_distortion = []
        for plane in range(img_array.shape[0]):
            try:
                distortion = get_distortion(
                    img_array[plane], pxl_size_xy, spacing=spacing, grid_size=grid_size
                )
            except (ValueError, RuntimeError):
                distortion = None
                filepath = metadata.get("filepath", "<unknown>")
                logger.warning(
                    f"Distortion analysis failed for plane {plane} of image {filepath}"
                )

            results_distortion.append(distortion)

        if debug:
            logger.debug(
                f"Plane distortion: {str([x['max_amplitude'] if x is not None else 'INF' for x in results_distortion])}"
            )

        # pick best Z plane (smallest max_amplitude)
        valid_entries = [
            (i, r) for i, r in enumerate(results_distortion) if r is not None
        ]

        if not valid_entries:
            # total failure
            return {
                "figure": None,
                "figure name": "Distortion",
                "status": "error",
                "measurements": [
                    {
                        "category": "Distortion",
                        "name": "Amplitude",
                        "value": None,
                        "unit": "µm",
                    }
                ],
            }

        best_idx, best_result_distortion = min(
            valid_entries, key=lambda x: x[1]["max_amplitude"]
        )

    paired_lattice = best_result_distortion["paired_lattice"]
    points = best_result_distortion["points"]
    img = best_result_distortion["img"]

    deformation_vectors = paired_lattice - points  # Shape (441,2)

    # Compute magnitude and direction
    magnitude = np.linalg.norm(deformation_vectors, axis=1)  # Shape (441,)
    max_vector_amplitude = np.max(magnitude)

    # Max distortion along x and y
    max_distortion_x = np.max(np.abs(deformation_vectors[:, 0]))
    max_distortion_y = np.max(np.abs(deformation_vectors[:, 1]))
    max_distortion_x_percent = (max_distortion_x / grid_size) * 100
    max_distortion_y_percent = (max_distortion_y / grid_size) * 100

    # visualization
    fig = None
    if visu:
        n_points = points.shape[0]
        full_grid = n_points == grid_size * grid_size

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

        # --- left: original image + fitted lattice + detected centers ---
        ax1.imshow(img, origin="lower", cmap="gray")
        ax1.plot(
            paired_lattice[:, 0],
            paired_lattice[:, 1],
            "ro",
            markersize=2,
            markerfacecolor="none",
            label="Fitted lattice",
            alpha=0.7,
        )
        ax1.plot(
            points[:, 0],
            points[:, 1],
            "go",
            markersize=4,
            markerfacecolor="none",
            label="Detected centers",
            alpha=0.7,
        )
        ax1.set_title("Centered image and fitted lattice")
        ax1.axis("off")
        ax1.legend(loc="upper right", fontsize=8)

        # --- right: distortion map ---
        if full_grid:
            # We can safely reshape into (grid_size, grid_size)
            sorted_points, sort_indices = sort_grid(points, gridsize=grid_size)
            X = sorted_points[:, 0].reshape(grid_size, grid_size)
            Y = sorted_points[:, 1].reshape(grid_size, grid_size)

            final_image = magnitude[sort_indices].reshape(grid_size, grid_size)

            level = 3
            CS = ax2.contour(
                sorted_points[:, 1].reshape(grid_size, grid_size),
                sorted_points[:, 0].reshape(grid_size, grid_size),
                filters.gaussian(final_image, 0.5),
                levels=level,
                linewidths=range(level),
                colors="black",
            )
            ax2.clabel(CS, fontsize=10)

            im = ax2.imshow(
                final_image,
                extent=[X.min(), X.max(), Y.min(), Y.max()],
                cmap="jet",
                alpha=0.7,
                vmin=0,
                vmax=10,
                origin="lower",
            )

            # prepare for the vector visualization on a regular grid
            dx = (X.max() - X.min()) / (grid_size - 1)
            dy = (Y.max() - Y.min()) / (grid_size - 1)
            X_grid = np.linspace(X.min() + dx / 2, X.max() - dx / 2, grid_size)
            Y_grid = np.linspace(Y.min() + dy / 2, Y.max() - dy / 2, grid_size)
            X_centered, Y_centered = np.meshgrid(X_grid, Y_grid)

            ax2.quiver(
                X_centered,
                Y_centered,
                deformation_vectors[:, 0][sort_indices].reshape(grid_size, grid_size),
                deformation_vectors[:, 1][sort_indices].reshape(grid_size, grid_size),
                color="black",
                angles="xy",
                scale_units="xy",
                scale=0.1,
                pivot="middle",
                minlength=1,
            )

        else:
            # Fallback: arbitrary number of points; just scatter + quiver
            logger.info(
                f"[Distortion] Detected {n_points} points (expected {grid_size**2}); "
                "using scatter/quiver visualization instead of regular grid."
            )

            # color-coded by magnitude
            im = ax2.scatter(
                points[:, 0],
                points[:, 1],
                c=magnitude,
                cmap="jet",
                vmin=0,
                vmax=10,
                alpha=0.8,
                s=15,
            )

            ax2.quiver(
                points[:, 0],
                points[:, 1],
                deformation_vectors[:, 0],
                deformation_vectors[:, 1],
                color="black",
                angles="xy",
                scale_units="xy",
                scale=0.1,
                pivot="middle",
                minlength=1,
            )

        ax2.set_title("Microscope Field Distortion")
        ax2.axis("off")
        plt.colorbar(im, label="Deformation magnitude (pixels)", ax=ax2)

        fig.subplots_adjust(bottom=0.1, left=0.05, right=0.95)

        # Add statistics text at the bottom
        stats_text = (
            f"Max Distortion Along X: {max_distortion_x_percent:.2f}% - "
            f"Max Distortion Along Y: {max_distortion_y_percent:.2f}%\n"
            f"Maximum Vector Amplitude: {max_vector_amplitude * pxl_size_xy:.2f} µm "
            f"(N={n_points} points)"
        )

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

    # Extra measurements, Geometric distortion rate
    total_deformation = np.sum(magnitude)
    geometric_distortion_rate = (
        total_deformation / (grid_size * grid_size * grid_size)
    ) * 100

    valid = np.isfinite(max_vector_amplitude)
    return {
        "figure": fig if visu else None,
        "figure name": "Distortion",
        "status": "success" if valid else "error",
        "measurements": [
            {
                "category": "Distortion",
                "name": "Amplitude",
                "value": max_vector_amplitude * pxl_size_xy,
                "unit": "µm",
            },
        ],
        "extra_measurements": {
            "max_vector_amplitude_pxl": max_vector_amplitude,
            "geometric_distortion_rate": geometric_distortion_rate,
            "max_distortion_x_percent": max_distortion_x_percent,
            "max_distortion_y_percent": max_distortion_y_percent,
            "best_z_plane": best_idx,
        },
    }
