"""
src/METEOR_core/coregistration_analysis.py
Created on Fri Jun 13 14:43:25 2025

@author: simon

metadata needed:
- 'Physical pixel size X' (µm)
- 'Image Size C' (number of channels)
- 'Image Size Z' (number of Z planes)
"""

# general import
from typing import Any

# analysis import
import matplotlib.pyplot as plt
import numpy as np
from skimage import filters, measure, morphology

# local import
from . import logger
from .focus import select_focus_plane
from .functions import (
    dimension_slicer,
    get_centroid,
    norm_img,
)
from .ring_lattice import detect_ring_lattice, match_points


def get_coregistration(
    img: np.ndarray,
    pxl_size_xy: float,
    spacing: float = 4.9,
    grid_size: int = 21,
    debug: bool = False,
) -> dict[str, Any]:
    blue = img[0]
    red = img[1]
    # median filter first
    blue = filters.median(blue, footprint=morphology.disk(1))
    red = filters.median(red, footprint=morphology.disk(1))
    # normalize the image
    blue = norm_img(blue)
    red = norm_img(red)
    blue_c, red_c = blue, red
    registration = detect_ring_lattice(blue_c, spacing / pxl_size_xy, grid_size)
    red_points = get_centroid(
        red_c, None, thr=registration.diagnostics["detection_threshold"], verbose=debug
    )
    if red_points is None or len(red_points) < 9:
        raise ValueError("Failed to detect sufficient red channel points")
    pairs = match_points(
        registration.points_xy, red_points, 0.35 * registration.spacing_px
    )
    if len(pairs) < 9:
        raise ValueError("Insufficient matched rings across channels")
    blue_points = registration.points_xy[pairs[:, 0]]
    red_points = red_points[pairs[:, 1]]

    return {
        "max_amplitude": np.max(np.linalg.norm(blue_points - red_points, axis=1)),  # type: ignore
        "red_c": red_c,
        "blue_c": blue_c,
        "blue_points": blue_points,
        "red_points": red_points,
        "registration": registration.diagnostics,
        "channel_matches": int(len(pairs)),
    }


def coregistration_analysis(
    img_array: np.ndarray,
    metadata: dict[str, Any],
    grid_size: int = 21,
    spacing: float = 5.1,
    visu: bool = True,
    debug: bool = False,
    dim: str = "CZYX",
) -> dict[str, Any]:
    """
    Compute lateral co-registration accuracy between two channels via a 2d rings target.
    Requires at least two channels.
    Parameters
    ----------
    img_array : np.ndarray
        Image array with at least two channels and multiple Z planes, shape CZYX.
    grid_size : int
        Number of points in the grid (e.g., 21 for a 21x21 grid).
    spacing : float
        Spacing between points in microns.
    metadata : dict[str, Any]
        Metadata dictionary containing at least 'Physical pixel size X' (µm),
        'Image Size C' (number of channels) and 'Image Size Z' (number of Z planes)
    visu : bool, optional
        Whether to generate a visualization figure. The default is True.
    debug : bool, optional
        Whether to enable debug logging. The default is False.
    Returns
    -------
    dict[str, Any]
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

    if int(metadata.get("Image Size C", 0)) < 2:
        raise ValueError(
            f"Bad shape, image need at least 2 colors, detected {int(metadata.get('Image Size C', 0))}"
        )

    if int(metadata.get("Image Size Z", 0)) < 2:
        raise ValueError(
            f"Bad shape, image need be of multiple Z, detected {int(metadata.get('Image Size Z', 0))}"
        )

    img_array = dimension_slicer(img_array, metadata, dim)

    try:
        best_idx, focus_scores = select_focus_plane(img_array[0])
        best_results_coregistration = get_coregistration(
            img_array[:, best_idx],
            pxl_size_xy,
            spacing=spacing,
            grid_size=grid_size,
            debug=debug,
        )
    except (ValueError, IndexError) as exc:
        logger.warning(f"Coregistration focus selection or analysis failed: {exc}")
        best_results_coregistration = None
    if best_results_coregistration is None:
        logger.error("Coregistration analysis failed on all planes.")
        return {
            "figure": None,
            "figure name": "Coregistration",
            "status": "error",
            "measurements": [
                {
                    "category": "Coregistration",
                    "name": "Pearson",
                    "value": None,
                    "unit": "",
                },
                {
                    "category": "Coregistration",
                    "name": "Amplitude",
                    "value": None,
                    "unit": "µm",
                },
            ],
            "extra_measurements": {},
        }

    red_c = best_results_coregistration["red_c"]
    blue_c = best_results_coregistration["blue_c"]
    red_points = best_results_coregistration["red_points"]
    blue_points = best_results_coregistration["blue_points"]
    deformation_vectors = blue_points - red_points

    # Compute magnitude and direction
    magnitude = np.linalg.norm(
        deformation_vectors, axis=1
    )  # Shape (grid_size*grid_size,)
    max_vector_amplitude = np.max(magnitude)
    # Pearson
    pearson_coeff, _ = measure.pearson_corr_coeff(red_c, blue_c)

    # make the figure
    fig = None
    if visu:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))  # type: ignore
        rgb = np.stack([red_c, blue_c, np.zeros(red_c.shape)], axis=-1)
        ax1.imshow(rgb)
        ax1.scatter(blue_points[:, 0], blue_points[:, 1], s=5, c="cyan")
        ax1.set_title("Image and matched ring centers")
        ax1.axis("off")
        im = ax2.scatter(
            blue_points[:, 0], blue_points[:, 1], c=magnitude, cmap="jet", s=12
        )
        ax2.quiver(
            blue_points[:, 0],
            blue_points[:, 1],
            deformation_vectors[:, 0],
            deformation_vectors[:, 1],
            color="black",
            angles="xy",
            scale_units="xy",
            scale=0.1,
            pivot="middle",
            minlength=1,
        )

        ax2.set_title("Lateral co-registration accuracy")
        ax2.invert_yaxis()
        ax2.set_aspect("equal")
        ax2.axis("off")

        plt.colorbar(im, label="Co-registration Magnitude (pixel)")

        fig.subplots_adjust(
            bottom=0.1, left=0.05, right=0.95
        )  # Ensure enough space for both plots and the text
        # Add statistics text at the bottom
        stats_text = (
            f"Pearson coefficient: {pearson_coeff:.4f} - "
            f"Maximum Vector Amplitude: {max_vector_amplitude:.2f} pxl - "
            f"Maximum Vector Amplitude: {max_vector_amplitude * pxl_size_xy:.2f} microns"
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

        plt.tight_layout()
        if __name__ == "__main__":
            plt.show()

    valid = (pearson_coeff is not None) and np.isfinite(max_vector_amplitude)
    return {
        "figure": fig if visu else None,
        "figure name": "Coregistration",
        "status": "success" if valid else "error",
        "measurements": [
            {
                "category": "Coregistration",
                "name": "Pearson",
                "value": pearson_coeff,
                "unit": "",
            },
            {
                "category": "Coregistration",
                "name": "Amplitude",
                "value": max_vector_amplitude * pxl_size_xy,
                "unit": "µm",
            },
        ],
        "extra_measurements": {
            "best_z_plane": best_idx,
            "focus_scores": focus_scores,
            "focus_at_stack_boundary": len(focus_scores) > 1
            and best_idx in (0, len(focus_scores) - 1),
            "pearson_coeff": pearson_coeff,
            "max_vector_amplitude_pxl": max_vector_amplitude,
            "ring_registration": best_results_coregistration["registration"],
            "matched_ring_count": best_results_coregistration["channel_matches"],
            "median_amplitude_um": float(np.median(magnitude) * pxl_size_xy),
            "p95_amplitude_um": float(np.percentile(magnitude, 95) * pxl_size_xy),
        },
    }
