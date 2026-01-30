"""
Created on Mon Jun 16 09:54:15 2025

@author: simon
"""

# general import
import multiprocessing
import random

# analysis import
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib.patches import Ellipse
from scipy import ndimage
from skimage import filters, measure, morphology, transform

# local import
from . import logger
from .functions import dimension_slicer, list_multiply, norm_img, round_to_even


def log_scale(image):  # log the image
    return np.log1p(image * 500)


def rescale_z_projection(image, z_scale_factor):  # Visualization purpose
    return transform.rescale(image, (z_scale_factor, 1), anti_aliasing=True)


def calculate_crop_sizes(bead_size, metadata, wavelength=0.5, multi=4):
    # Estimate PSF dimensions based on optics
    psf_z = wavelength / (float(metadata["Lens NA"]) ** 1.45)
    psf_xy = wavelength / (1.45 * float(metadata["Lens NA"]))

    # Set cropping bounds based on PSF and bead size
    crop_size_microns = {
        "X": max(multi * psf_xy, bead_size * 1.5),
        "Y": max(multi * psf_xy, bead_size * 1.5),
        "Z": max(multi * psf_z, bead_size * 2),
    }

    large_crop_microns = {
        "X": 2 * crop_size_microns["X"],
        "Y": 2 * crop_size_microns["Y"],
        "Z": 2 * crop_size_microns["Z"],
    }

    # Ensure requested crop sizes don’t exceed image size limits
    for dim in ["X", "Y", "Z"]:
        max_available_microns = (
            float(metadata["Image Size " + dim])
            * float(metadata["Physical pixel size " + dim])
        ) - 0.8

        # max_available_microns = (image_size[dim] * physical_pixel_sizes[dim]) -.8
        if large_crop_microns[dim] > max_available_microns:
            large_crop_microns[dim] = max_available_microns
        if crop_size_microns[dim] > max_available_microns:
            crop_size_microns[dim] = max_available_microns

    return large_crop_microns, crop_size_microns


def adjust_crop_bounds(
    min_val, max_val, target_size, img_size
):  # cropping of the beads
    if max_val > img_size:
        max_val = img_size
        min_val = max(0, img_size - target_size)
    elif min_val < 0:
        min_val = 0
        max_val = min(img_size, target_size)

    # Ensure the crop size is exactly as specified
    current_size = max_val - min_val
    if current_size < target_size:
        if max_val < img_size:
            max_val = min(img_size, max_val + (target_size - current_size))
        else:
            min_val = max(0, min_val - (target_size - current_size))
    elif current_size > target_size:
        max_val = min_val + target_size

    return min_val, max_val


def process_beads(
    img: np.ndarray,
    min_size: int,
    max_bead_size: int,
    visu: bool = False,
    verbose: bool = False,
) -> tuple[np.ndarray, list]:
    """
    Segment 3D beads from a PSF stack and filter by size.

    Parameters
    ----------
    img : np.ndarray
        3D image stack (Z, Y, X), normalized upstream.
    min_size : int
        Minimum volume (in voxels) to keep a bead (for remove_small_objects).
    max_bead_size : int
        Maximum allowed bead volume (in voxels) to keep (filtration).
    visu : bool
        If True, display a max-projection with bead centroids.
    verbose : bool
        If True, log extra information.

    Returns
    -------
    labeled_img : np.ndarray
        3D labeled image of connected components.
    filtered_beads : list[skimage.measure._regionprops.RegionProperties]
        Filtered regionprops (each bead).
    """
    # Apply 3D Otsu thresholding
    thresh = filters.threshold_otsu(img)
    binary = img > thresh

    if np.sum(binary) > list_multiply(img.shape) * 0.1:  # 10% of volume
        if verbose:
            logger.info(
                "[process_beads]: Otsu too permissive, switching to Triangle threshold."
            )
        thresh = filters.threshold_triangle(img)
        binary = img > thresh

    # 3D morphological operations to remove noise and fill holes
    binary = morphology.remove_small_objects(binary, min_size=min_size)
    binary = morphology.binary_closing(binary, morphology.ball(2))

    # Label 3D connected components
    labeled_img = measure.label(binary)
    regions = measure.regionprops(labeled_img, img)

    # Filter beads based on 3D size
    if verbose:
        logger.info("[process_beads]: Filter beads based on their size")

    filtered_beads = [r for r in regions if r.area <= max_bead_size]

    if verbose:
        logger.info(f"[process_beads]: Number of detected beads: {len(filtered_beads)}")

    random.shuffle(filtered_beads)  # shuffle them

    if visu and filtered_beads and verbose:
        bead_centroids = np.array([r.centroid for r in filtered_beads])
        plt.figure(figsize=(10, 10))
        plt.imshow(np.max(img, axis=0))
        plt.scatter(bead_centroids[:, 2], bead_centroids[:, 1], s=1, c="r", alpha=0.5)
        plt.title("Potential bead location")
        plt.axis("off")
        plt.show()

    return np.array(labeled_img), filtered_beads


def fit_ellipse_2d(
    psf_slice: np.ndarray,
    pixel_size_x: float,
    pixel_size_y: float,
    title: str,
    ax,
    visu: bool = True,
) -> tuple[float, ...] | tuple[None, None, None]:
    """
    Fit an ellipse to the half-maximum contour of a 2D PSF slice and extract
    FWHM-like measures and tilt.

    Parameters
    ----------
    psf_slice : np.ndarray
        2D slice of the PSF, ideally normalized (0..1).
    pixel_size_x : float
        Physical pixel size along X (µm).
    pixel_size_y : float
        Physical pixel size along Y or Z (µm).
    title : str
        Title for the plot (used to decide axis labels).
    ax : matplotlib.axes.Axes or None
        Axis to draw on if visu=True. If None, no drawing is done.
    visu : bool
        If True, show the image, contours and fitted ellipse.

    Returns
    -------
    fwhm_x : float or None
        Effective FWHM-like measure along X (µm), or None if fit fails.
    fwhm_y : float or None
        Effective FWHM-like measure along Y (µm), or None if fit fails.
    theta_deg : float or None
        Ellipse orientation angle in degrees, or None if fit fails.
    """
    # Half-max contour
    contours = measure.find_contours(psf_slice, 0.5)
    if not contours:
        logger.debug(f"[fit_ellipse_2d]: No valid contour found for {title}")
        return None, None, None

    largest_contour = max(contours, key=len)
    # make it the correct size
    largest_contour[:, 0] = largest_contour[:, 0] * pixel_size_y
    largest_contour[:, 1] = largest_contour[:, 1] * pixel_size_x

    ellipse = measure.EllipseModel()
    if not ellipse.estimate(largest_contour):
        logger.debug(f"[fit_ellipse_2d]: Ellipse fit failed for {title}")
        return None, None, None

    yc, xc, a, b, theta = ellipse.params
    fwhm_major = 2 * max(a, b)
    fwhm_minor = 2 * min(a, b)
    theta_deg = np.degrees(theta)

    # Project ellipse axes onto X/Y directions
    fwhm_x = fwhm_major * np.cos(theta) + fwhm_minor * np.sin(theta)
    fwhm_y = fwhm_major * np.sin(theta) + fwhm_minor * np.cos(theta)

    if not visu or ax is None:
        return abs(fwhm_x), abs(fwhm_y), theta_deg

    # Plot the PSF slice and fitted ellipse
    ax.imshow(
        psf_slice,
        extent=[
            0,
            psf_slice.shape[1] * pixel_size_x,
            0,
            psf_slice.shape[0] * pixel_size_y,
        ],
        cmap="gray",
        origin="lower",
    )
    ax.plot(
        largest_contour[:, 1], largest_contour[:, 0], "r--", label="Half-Max Contour"
    )
    # Optional extra contours (0.15 and 0.9), if they exist
    for level, style in [(0.15, "g--"), (0.9, "b--")]:
        extra_contours = measure.find_contours(psf_slice, level)
        if extra_contours:
            c = extra_contours[0].copy()
            c[:, 0] *= pixel_size_y
            c[:, 1] *= pixel_size_x
            ax.plot(c[:, 1], c[:, 0], style, alpha=0.7)

    # Draw the fitted ellipse
    ellipse_patch = Ellipse(
        (xc, yc),
        zorder=100,
        width=2 * a,
        height=2 * b,
        angle=theta_deg + 90,
        edgecolor="blue",
        facecolor="none",
        linewidth=2,
        linestyle=":",
        label="Fitted Ellipse",
    )
    ax.add_patch(ellipse_patch)

    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Z (µm)" if "Z" in title else "Y (µm)")
    ax.set_title(f"{title}")

    return abs(fwhm_x), abs(fwhm_y), theta_deg


def process_bead(
    centroid: tuple,
    img: np.ndarray,
    labeled_img: np.ndarray,
    large_crop_size: dict,
    crop_size_pixels: dict,
    ignore_edge: bool,
    upsample_factor: int,
) -> np.ndarray | None:
    """
    Process a single bead to extract a normalized PSF subvolume.

    Steps:
    - Take a coarse 3D crop around the bead centroid (large_crop_size).
    - Optionally reject beads too close to the image edges (ignore_edge).
    - Ensure only one labeled object is present in the coarse crop.
    - Upsample the coarse crop isotropically by upsample_factor.
    - Re-detect the bead in the upsampled label and crop a tighter PSF box
      (crop_size_pixels) around the upsampled centroid.
    - Normalize the PSF so that its sum is 1.
    - Reject weak detections (low peak vs mean+std).

    Parameters
    ----------
    centroid : tuple[float, float, float]
        Bead centroid in (z, y, x) coordinates (in pixel units of original image).
    img : np.ndarray
        Original 3D image stack (Z, Y, X).
    labeled_img : np.ndarray
        3D labeled image of connected components matching `img`.
    large_crop_size : dict
        {'Z', 'Y', 'X'} coarse crop size in pixels (original resolution).
    crop_size_pixels : dict
        {'Z', 'Y', 'X'} tight PSF crop size in pixels (upsampled resolution).
    ignore_edge : bool
        If True, beads too close to volume borders are discarded.
    upsample_factor : int
        Isotropic upsampling factor applied to the coarse crop.

    Returns
    -------
    psf : np.ndarray or None
        PSF subvolume (Z, Y, X) at upsampled resolution, normalized to sum=1,
        or None if this bead is rejected at any stage.
    """
    try:
        z, y, x = centroid

        # Coarse crop bounds in original pixel coordinates
        z_start = int(z - large_crop_size["Z"] // 2)
        z_end = int(z + large_crop_size["Z"] // 2)
        y_start = int(y - large_crop_size["Y"] // 2)
        y_end = int(y + large_crop_size["Y"] // 2)
        x_start = int(x - large_crop_size["X"] // 2)
        x_end = int(x + large_crop_size["X"] // 2)

        # Reject beads near edges if requested
        if ignore_edge:
            for start, end, dim in zip(
                (z_start, y_start, x_start),
                (z_end, y_end, x_end),
                img.shape,
                strict=False,
            ):
                if start <= 0 or end >= dim:
                    return None

        # Ensure coarse crop has exactly the requested size
        if (
            (z_end - z_start) != large_crop_size["Z"]
            or (y_end - y_start) != large_crop_size["Y"]
            or (x_end - x_start) != large_crop_size["X"]
        ):
            return None

        large_crop = img[z_start:z_end, y_start:y_end, x_start:x_end]
        large_crop_label = labeled_img[z_start:z_end, y_start:y_end, x_start:x_end]

        if len(np.unique(large_crop_label)) != 2:  # backgound and bead
            return None  # More than one bead

        upsampled_crop = ndimage.zoom(large_crop, (upsample_factor,) * 3, order=0)
        upsampled_label = (
            ndimage.zoom(large_crop_label, (upsample_factor,) * 3, order=3) > 0
        )

        subregions = measure.regionprops(
            upsampled_label.astype(np.uint8), intensity_image=upsampled_crop
        )
        if not subregions:
            return None

        # Use largest region
        region = max(subregions, key=lambda r: r.area)
        z0, y0, x0 = region.centroid
        z_size, y_size, x_size = upsampled_crop.shape

        def adjust_crop_bounds(min_val, max_val, target_size, img_size):
            # Clip to valid range
            if max_val > img_size:
                max_val = img_size
                min_val = max(0, img_size - target_size)
            elif min_val < 0:
                min_val = 0
                max_val = min(img_size, target_size)

            current_size = max_val - min_val
            if current_size < target_size:
                if max_val < img_size:
                    max_val = min(img_size, max_val + (target_size - current_size))
                else:
                    min_val = max(0, min_val - (target_size - current_size))
            elif current_size > target_size:
                max_val = min_val + target_size

            return min_val, max_val

        x_min, x_max = adjust_crop_bounds(
            int(x0 - crop_size_pixels["X"] // 2),
            int(x0 + crop_size_pixels["X"] // 2),
            crop_size_pixels["X"],
            x_size,
        )
        y_min, y_max = adjust_crop_bounds(
            int(y0 - crop_size_pixels["Y"] // 2),
            int(y0 + crop_size_pixels["Y"] // 2),
            crop_size_pixels["Y"],
            y_size,
        )
        z_min, z_max = adjust_crop_bounds(
            int(z0 - crop_size_pixels["Z"] // 2),
            int(z0 + crop_size_pixels["Z"] // 2),
            crop_size_pixels["Z"],
            z_size,
        )

        psf = upsampled_crop[z_min:z_max, y_min:y_max, x_min:x_max]

        # Ensure right shape; if not, discard
        expected_shape = (
            crop_size_pixels["Z"],
            crop_size_pixels["Y"],
            crop_size_pixels["X"],
        )
        if psf.shape != expected_shape:
            return None

        psf_sum = np.sum(psf)
        if psf_sum == 0:
            return None
        psf = (psf / psf_sum).astype(np.float32)

        if np.max(psf) < np.mean(psf) + np.std(psf) * 1:
            return None  # Weak detection

        return psf

    except (ValueError, IndexError, TypeError) as e:
        logger.debug(f"[process_bead]: exception for centroid {centroid}: {e}")
        return None


def bead_extraction(
    img: np.ndarray,
    labeled_img: np.ndarray,
    upsample_factor: int,
    filtered_beads: list,
    crop_size_microns: dict,
    large_crop_microns: dict,
    physical_pixel_sizes: dict,
    visu: bool = True,
    verbose: bool = True,
    ignore_edge: bool = True,
):
    """
    Extract individual PSFs from bead centroids, upsample, center, crop and
    average them.

    Parameters
    ----------
    img : np.ndarray
        3D stack (Z, Y, X).
    labeled_img : np.ndarray
        3D labeled image corresponding to `img`.
    upsample_factor : int
        Isotropic upsampling factor for better peak localization.
    filtered_beads : list
        List of regionprops (beads) from `process_beads`.
    crop_size_microns : dict
        {'X', 'Y', 'Z'} crop size (inner PSF box) in µm.
    large_crop_microns : dict
        {'X', 'Y', 'Z'} larger crop used before upsampling (in µm).
    physical_pixel_sizes : dict
        {'X', 'Y', 'Z'} physical pixel sizes in µm.
    visu : bool
        If True, call `bead_extraction_visu` for a quick visual check.
    verbose : bool
        If True, log progress.
    ignore_edge : bool
        If True, discard beads too close to the border.

    Returns
    -------
    average_psf : np.ndarray or None
        Averaged PSF, shape (Z, Y, X) at upsampled resolution, or None if no bead.
    psf_array : np.ndarray or None
        Stack of individual PSFs, shape (N, Z, Y, X), or None if no bead.
    """
    # Convert µm → pixels and enforce even sizes
    crop_size = {
        "X": round_to_even(
            crop_size_microns["X"] / physical_pixel_sizes["X"] * upsample_factor
        ),
        "Y": round_to_even(
            crop_size_microns["Y"] / physical_pixel_sizes["Y"] * upsample_factor
        ),
        "Z": round_to_even(
            crop_size_microns["Z"] / physical_pixel_sizes["Z"] * upsample_factor
        ),
    }
    large_crop_size = {
        "X": round_to_even(large_crop_microns["X"] / physical_pixel_sizes["X"]),
        "Y": round_to_even(large_crop_microns["Y"] / physical_pixel_sizes["Y"]),
        "Z": round_to_even(large_crop_microns["Z"] / physical_pixel_sizes["Z"]),
    }

    centroids = [bead.centroid for bead in filtered_beads]

    # initialization
    valid_psfs = []
    target_psfs = 100
    buffer_factor = 1.2
    num_processors = multiprocessing.cpu_count()
    batch_size = 100
    start = 0

    while len(valid_psfs) < 100 and start < len(centroids):
        end = min(start + batch_size, len(centroids))
        batch = centroids[start:end]
        results = Parallel(n_jobs=-1, verbose=0)(
            delayed(process_bead)(
                centroid,
                img,
                labeled_img,
                large_crop_size,
                crop_size,
                ignore_edge,
                upsample_factor,
            )
            for centroid in batch
        )
        valid_psfs.extend([psf for psf in results if psf is not None])
        start += batch_size

        # Adjust batch size based on remaining needed PSFs
        remaining_needed = max(0, target_psfs - len(valid_psfs))
        batch_size = min(
            100, max(num_processors, int(remaining_needed * buffer_factor))
        )

    if len(valid_psfs) == 0:
        if verbose:
            logger.warning("[bead_extraction]: no valid PSFs extracted from beads.")
        return None, None

    psf_array = np.asarray(valid_psfs, dtype=np.float32)
    if verbose:
        logger.info(f"[bead_extraction]: Collected {len(psf_array)} valid PSFs.")

    average_psf = np.mean(psf_array, axis=0)

    if visu and verbose:
        bead_extraction_visu(psf_array, average_psf, physical_pixel_sizes)

    return average_psf, psf_array


def bead_extraction_visu(psf_array, average_psf, physical_pixel_sizes: dict):
    # Define Z scaling factor
    z_scale_factor = physical_pixel_sizes["Z"] / physical_pixel_sizes["X"]
    # get a random bead
    rd_psf = psf_array[np.random.choice(len(psf_array))]
    # Apply log scaling and rescale Z projections
    projections = {
        "RAND - Max Projection (XY)": log_scale(np.max(rd_psf, axis=0)),
        "RAND - Max Projection (XZ)": rescale_z_projection(
            log_scale(np.max(rd_psf, axis=1)), z_scale_factor
        ),
        "RAND - Max Projection (YZ)": rescale_z_projection(
            log_scale(np.max(rd_psf, axis=2)), z_scale_factor
        ),
        "AVG - Max Projection (XY)": log_scale(np.max(average_psf, axis=0)),
        "AVG - Max Projection (XZ)": rescale_z_projection(
            log_scale(np.max(average_psf, axis=1)), z_scale_factor
        ),
        "AVG - Max Projection (YZ)": rescale_z_projection(
            log_scale(np.max(average_psf, axis=2)), z_scale_factor
        ),
    }

    # Plot
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))

    for i, (title, im) in enumerate(projections.items()):
        ax[i // 3, i % 3].imshow(im, cmap="viridis", interpolation="none")
        ax[i // 3, i % 3].set_title(title)
        ax[i // 3, i % 3].axis("off")  # Remove axes for better visualization

    plt.tight_layout()
    plt.show()


def return_error() -> dict:
    return {
        "figure": None,
        "figure name": "PSF_beads",
        "status": "error",
        "measurements": [
            {"category": "PSF", "name": "PSF X", "value": None, "unit": "µm"},
            {"category": "PSF", "name": "PSF Y", "value": None, "unit": "µm"},
            {"category": "PSF", "name": "PSF Z", "value": None, "unit": "µm"},
        ],
        "extra_measurements": {
            "num_beads_used": 0,
        },
        "gif": None,
        "tiff": None,
    }


def psf_analysis(
    img_array: np.ndarray,
    metadata: dict,
    bead_size: float = 0.3,
    ignore_edge: bool = False,
    debug: bool = False,
    visu: bool = False,
    dim="ZYX",
) -> dict:
    """
    PSF analysis from sub-resolution beads.

    Workflow:
    - Normalize the 3D stack.
    - Detect beads in 3D and filter by volume.
    - Define cropping / upsampling parameters (based on NA, bead size, etc.).
    - Extract and average individual PSFs.
    - Fit ellipses on XY, XZ, YZ half-maximum contours to estimate FWHM in X, Y, Z.
    - Package results for the metrology pipeline, including optional data
      to build GIFs and OME-TIFF outside this function.

    Parameters
    ----------
    img_array : np.ndarray
        3D image stack (Z, Y, X), ideally a bead field.
    bead_size : float
        Bead diameter or radius in µm (depending on your convention; here it is
        used to estimate bead volume).
    metadata : dict
        Must contain at least:
            - 'Physical pixel size X'
            - 'Physical pixel size Y'
            - 'Physical pixel size Z'
            - 'Lens NA'
            - 'Image Size X', 'Image Size Y', 'Image Size Z'
    verbose : bool
        If True, log extra information.

    Returns
    -------
    dict
        Metrology result dict:

        {
            "figure": Figure or None,
            "figure name": "PSF_beads",
            "status": "success" or "error",
            "measurements": [...],
            "extra_measurements": {...},
            "gif": {...},
            "tiff": {...},
        }
    """

    img_array = dimension_slicer(img_array, metadata, dim)
    if (
        int(metadata.get("Image Size C", 0)) > 1
    ):  # security, in case of multi-C, select the first one
        img_array = img_array[:, :1, ...]

    # normalize the image
    img_array = norm_img(img_array)
    # Process the image (threshold and size cleaning)
    if debug:
        logger.info(
            f"[psf_analysis]: Processing the image with shape {img_array.shape}"
        )

    physical_pixel_sizes = {
        "X": float(metadata.get("Physical pixel size X", 1.0)),
        "Y": float(metadata.get("Physical pixel size Y", 1.0)),
        "Z": float(metadata.get("Physical pixel size Z", 1.0)),
    }

    # Approximate bead volume (sphere)
    bead_volume = (4 / 3) * np.pi * (bead_size) ** 3
    bead_volume_pixels = int(
        bead_volume / list_multiply(list(physical_pixel_sizes.values()))
    )

    # Bead segmentation
    labeled_img, filtered_beads = process_beads(
        img_array,
        int(bead_volume_pixels * 0.2),
        int(bead_volume_pixels * 1.5),
        verbose=debug,
        visu=visu,
    )

    if len(filtered_beads) == 0:
        if debug:
            logger.warning("[psf_analysis]: no beads detected after filtering.")
        return return_error()

    # cropping and centering bounding box size in microns
    large_crop_microns, crop_size_microns = calculate_crop_sizes(bead_size, metadata)
    if debug:
        logger.info("[psf_analysis]: Large Crop: " + str(large_crop_microns))
        logger.info("[psf_analysis]: Crop Size: " + str(crop_size_microns))

    upsample_factor = 10  # upsample to determine the subpixel localization of the bead
    average_psf, psf_array = bead_extraction(
        img_array,
        labeled_img,
        upsample_factor,
        filtered_beads,
        crop_size_microns,
        large_crop_microns,
        physical_pixel_sizes,
        verbose=debug,
        visu=visu,
        ignore_edge=ignore_edge,
    )

    if average_psf is None or psf_array is None:
        if debug:
            logger.warning("[psf_analysis]: bead_extraction returned no valid PSFs.")
        return return_error()

    upscaled_pxl_size = {
        x: physical_pixel_sizes[x] / upsample_factor for x in physical_pixel_sizes
    }

    # Extract mid-slices
    mid_z, mid_y, mid_x = np.array(average_psf.shape) // 2
    xy_slice = norm_img(average_psf[mid_z])  # XY plane
    xz_slice = norm_img(average_psf[:, mid_y, :])  # XZ plane
    yz_slice = norm_img(average_psf[:, :, mid_x])  # YZ plane

    # Set up figure if requested
    if visu:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    else:
        fig, ax1, ax2, ax3 = None, None, None, None
    # Fit ellipses
    fwhm_xy_major, fwhm_xy_minor, theta_xy = fit_ellipse_2d(
        xy_slice,
        upscaled_pxl_size["X"],
        upscaled_pxl_size["Y"],
        "XY Plane",
        ax1,
        visu=visu,
    )
    fwhm_xz_major, fwhm_xz_minor, theta_xz = fit_ellipse_2d(
        xz_slice,
        upscaled_pxl_size["X"],
        upscaled_pxl_size["Z"],
        "XZ Plane",
        ax2,
        visu=visu,
    )
    fwhm_yz_major, fwhm_yz_minor, theta_yz = fit_ellipse_2d(
        yz_slice,
        upscaled_pxl_size["Y"],
        upscaled_pxl_size["Z"],
        "YZ Plane",
        ax3,
        visu=visu,
    )

    # Check validity of ellipse fits
    if any(
        v is None
        for v in (
            fwhm_xy_major,
            fwhm_xy_minor,
            fwhm_xz_major,
            fwhm_xz_minor,
            fwhm_yz_major,
            fwhm_yz_minor,
            theta_xy,
            theta_xz,
            theta_yz,
        )
    ):
        if debug:
            logger.warning(
                "[psf_analysis]: one or more ellipse fits failed; cannot compute PSF sizes."
            )
        x_psf = y_psf = z_psf = None
        status = "error"
    else:
        # Normalize angles to [0, 90]
        for _, theta in [("XY", theta_xy), ("XZ", theta_xz), ("YZ", theta_yz)]:
            if theta and theta > 90:
                theta = abs(180 - theta)

        # Use your averaging scheme for PSF width, handling None values
        def safe_mean(a, b):
            if a is not None and b is not None:
                return (a + b) / 2.0
            elif a is not None:
                return a
            elif b is not None:
                return b
            else:
                return None

        x_psf = safe_mean(fwhm_xy_minor, fwhm_xz_minor)
        y_psf = safe_mean(fwhm_xy_major, fwhm_yz_minor)
        z_psf = safe_mean(fwhm_xz_major, fwhm_yz_major)
        status = "success"

        if debug:
            logger.info(
                f"[psf_analysis]: Measured 2D FWHM: X={x_psf:.3f} µm, Y={y_psf:.3f} µm, Z={z_psf:.3f} µm"
            )

    if visu and fig is not None:
        plt.tight_layout()
        if __name__ == "__main__":
            plt.show()

    num_beads_used = int(psf_array.shape[0])
    return {
        "figure": fig if visu else None,
        "figure name": "PSF_beads",
        "status": status,
        "measurements": [
            {"category": "PSF", "name": "PSF X", "value": x_psf, "unit": "µm"},
            {"category": "PSF", "name": "PSF Y", "value": y_psf, "unit": "µm"},
            {"category": "PSF", "name": "PSF Z", "value": z_psf, "unit": "µm"},
            {
                "category": "PSF",
                "name": "Angle XY",
                "value": theta_xy,
                "unit": "degree",
            },
            {
                "category": "PSF",
                "name": "Angle XZ",
                "value": theta_xz,
                "unit": "degree",
            },
            {
                "category": "PSF",
                "name": "Angle YZ",
                "value": theta_yz,
                "unit": "degree",
            },
        ],
        "extra_measurements": {
            "num_beads_used": num_beads_used,
            "average_psf_shape": average_psf.shape,
        },
        "gif": {
            "frames": psf_array,  # shape (N, Z, Y, X)
            "upscaled_pixel_size": upscaled_pxl_size,
            "name": "PSF_beads_evolution",
            "fps": 10,
        },
        "tiff": {
            "image": average_psf,
            "upscaled_pixel_size": upscaled_pxl_size,
            "name": "PSF_kernel",
        },
    }
