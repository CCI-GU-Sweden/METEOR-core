"""
src/METEOR_core/plastic_uniformity.py

Created on Mon Jun 16 12:19:45 2025

@author: simon

metadata needed:
- 'Physical pixel size X'
"""

# analysis import
import matplotlib.pyplot as plt
import numpy as np
from skimage import filters, transform
from skimage.draw import line

# local import
from . import logger
from .functions import centuring_acc, dimension_slicer, uniformity


def ensure_square(img):
    """Resize image to a square by cropping or padding."""
    height, width = img.shape
    if height == width:
        return img  # Already square
    size = min(height, width)
    return transform.resize(img[:size, :size], (size, size), preserve_range=True)


def bin_image(img, grid_size):
    """Bin image into superpixels by resizing to grid_size x grid_size."""
    return transform.resize(img, (grid_size, grid_size), preserve_range=True)


def process_img(img_stack, gaussian):
    processed_images = []

    for img in img_stack:
        img_square = ensure_square(img)
        img = filters.gaussian(img_square, gaussian)
        processed_images.append(img)

    final_image = np.mean(np.stack(processed_images), axis=0)

    return final_image


def calculate_field_uniformity(
    img_stack: np.ndarray,
    metadata: dict,
    img0=None,
    visu: bool = True,
    debug: bool = False,
    dim: str = "ZYX",
) -> dict:
    """
    Compute field illumination uniformity from a plastic slide.

    Workflow:
    - Optionally average multiple images (img_stack).
    - Optionally subtract a background reference stack (img0).
    - Gaussian smooth the result and bin into a coarse grid.
    - Normalize to percentage of maximum.
    - Compute:
        * Field uniformity = 100 * I_min / I_max (via `uniformity` helper).
        * Roll-off along main and anti-diagonals.
        * Centering accuracy (how far the intensity-weighted centroid is
          from the geometric center).

    Parameters
    ----------
    img_stack : array-like
        Iterable of 2D images (Y, X) OR a 3D stack (N, Y, X). Each image is
        assumed to be in different physical coordinates.
    metadata : dict
        Must contain 'Physical pixel size X' in µm.
    img0 : array-like or None
        Optional background / reference stack of the same shape/type as img_stack.
        If provided, it is processed identically and subtracted (clipped at 0).
    visu : bool
        If True, create a Matplotlib figure with:
            - reference averaged image,
            - uniformity map,
            - line profiles.
    debug : bool
        If True, log additional debug info.

    Returns
    -------
    dict
        Metrology result dict with keys:
        - "figure": Matplotlib Figure or None
        - "figure name": 'Plastic_Field_Uniformity'
        - "status": 'success' or 'error'
        - "measurements": list of two entries
            * Centering accuracy (%)
            * Uniformity (%)
        - "extra_measurements": roll-off along diagonals
    """
    gaussian_blur = 5
    grid_size = 200
    pxl_size_xy = float(metadata.get("Physical pixel size X", 1.0))

    if (
        int(metadata.get("Image Size C", 0)) > 1
    ):  # security, in case of multi-C, select the first one
        img_stack = img_stack[:, :1, ...]
        if debug:
            logger.info(
                "calculate_field_uniformity: multi-C image, taking C=0 only, new shape: "
                + str(img_stack.shape)
            )
    if (
        int(metadata.get("Image Size T", 0)) > 1
    ):  # in case of multi time points, take the first one
        img_stack = img_stack[:1, ...]
        if debug:
            logger.debug(
                f"[calculate_field_uniformity]: taking first time point for analysis: shape now {img_stack.shape}"
            )

    img_stack = dimension_slicer(img_stack, metadata, dim)

    img_stack = np.asarray(img_stack)
    if img_stack.ndim == 2:
        # single image
        img_stack_list = [img_stack]
    elif img_stack.ndim == 3:
        # assume shape (N, Y, X)
        img_stack_list = [img_stack[i] for i in range(img_stack.shape[0])]
    else:
        raise ValueError(f"Unsupported img_stack shape: {img_stack.shape}")

    if debug:
        logger.debug(
            f"[calculate_field_uniformity]: using {len(img_stack_list)} images."
        )

    image = process_img(img_stack_list, gaussian_blur)

    # Optional background subtraction
    if img0 is not None:
        img0 = np.asarray(img0)
        if img0.ndim == 2:
            img0_list = [img0]
        elif img0.ndim == 3:
            img0_list = [img0[i] for i in range(img0.shape[0])]
        else:
            raise ValueError(f"Unsupported img0 shape: {img0.shape}")

        bg = process_img(img0_list, gaussian_blur)
        image = np.clip(image - bg, 0, None)

    # Bin to coarse grid
    final_image = bin_image(image, grid_size)

    # Normalize
    max_val = np.max(final_image)
    if max_val <= 0:
        logger.warning(
            "[calculate_field_uniformity]: max intensity is <= 0; cannot normalize."
        )
        field_uniformity = np.nan
    else:
        final_image = final_image * (100 / np.max(final_image))
        field_uniformity = uniformity(final_image)

    # profiles
    diagonal = np.diag(final_image)
    anti_dia = np.diag(np.fliplr(final_image))
    min_coord = np.unravel_index(
        np.argmin(final_image), final_image.shape
    )  # (row, col)
    max_coord = np.unravel_index(np.argmax(final_image), final_image.shape)
    rr, cc = line(min_coord[0], min_coord[1], max_coord[0], max_coord[1])
    line_profile = final_image[rr, cc]

    roll_off_dia = np.max(diagonal) - np.min(diagonal)
    roll_off_adi = np.max(anti_dia) - np.min(anti_dia)

    # Geometric vs intensity-weighted center
    geo_center = ((grid_size - 1) / 2, (grid_size - 1) / 2)
    x_coords, y_coords = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    total_intensity = np.sum(final_image)
    if total_intensity > 0:
        centroid_x = float(np.sum(x_coords * final_image) / total_intensity)
        centroid_y = float(np.sum(y_coords * final_image) / total_intensity)
        centroid = (centroid_x, centroid_y)
    else:
        centroid = (np.nan, np.nan)

    centering_accuracy = centuring_acc(final_image)

    # visualization
    fig = None
    if visu:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        ax1.imshow(image.T, origin="lower")
        ax1.set_title("Reference image")
        ax1.axis("off")

        level = 4
        # Create pixel coordinate meshgrid
        x_coords, y_coords = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
        # Flatten coordinate arrays for proper structuring
        points = np.column_stack((x_coords.flatten(), y_coords.flatten()))
        sort_indices = np.argsort(
            final_image.flatten()
        )  # Sort indices based on intensity
        X = points[:, 0][sort_indices].reshape(grid_size, grid_size)
        Y = points[:, 1][sort_indices].reshape(grid_size, grid_size)
        CS = ax2.contour(
            points[:, 1].reshape(grid_size, grid_size),
            points[:, 0].reshape(grid_size, grid_size),
            filters.gaussian(final_image, 2),
            levels=level,
            linewidths=range(level),
            colors="black",
        )
        ax2.clabel(CS, fontsize=10)

        im = ax2.imshow(
            final_image.T,
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            cmap="jet_r",
            alpha=0.7,
            vmin=0,
            vmax=np.max(final_image),
            origin="lower",
        )

        ax2.plot(centroid[0], centroid[1], "r+", markersize=10, zorder=100)
        ax2.plot(geo_center[0], geo_center[1], "k+", markersize=10)

        ax2.set_title("Field Uniformity")
        ax2.axis("off")
        plt.colorbar(im, label="Intensity (%)")

        # Compute the real-world distances for the x-axis
        bin_size_x = (img_stack[0].shape[1] * pxl_size_xy) / grid_size
        diagonal_distance = np.arange(len(diagonal)) * bin_size_x
        anti_dia_distance = np.arange(len(anti_dia)) * bin_size_x
        line_profile_distance = np.arange(len(line_profile)) * bin_size_x
        ax3.plot(diagonal_distance, diagonal, "r", linestyle="-", label="Main diagonal")
        ax3.plot(anti_dia_distance, anti_dia, "g", linestyle="-", label="Anti diagonal")
        ax3.plot(
            line_profile_distance,
            line_profile,
            "k",
            linestyle="-",
            label="Min-Max profile",
        )
        ax3.set_xlabel(f"Distance ({bin_size_x:.2f} μm per binned pixel)")
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

    valid = np.isfinite(field_uniformity) and np.isfinite(centering_accuracy)
    return {
        "figure": fig if visu else None,
        "figure name": "Plastic_Field_Uniformity",
        "status": "success" if valid else "error",
        "measurements": [
            {
                "category": "Uniformity - Plastic",
                "name": "Centering accuracy - Plastic",
                "value": centering_accuracy,
                "unit": "",
            },
            {
                "category": "Uniformity - Plastic",
                "name": "Uniformity",
                "value": field_uniformity,
                "unit": "",
            },
        ],
        "extra_measurements": {
            "roll_off_dia": roll_off_dia,
            "roll_off_adi": roll_off_adi,
        },
    }
