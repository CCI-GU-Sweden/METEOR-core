"""
src/METEOR_core/RSF.py

Created on Fri Aug  8 11:39:08 2025

@author: simon

RSF.py

Ring spread function
Reconstruct the average ring from a 2D rings target (Argolight 2D rings)
Metadata needed:
- 'Physical pixel size X'
- 'Physical pixel size Z'
- "axis_order"

"""

# general import
from typing import Any

# analysis import
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import filters, measure, transform

# local import
from . import logger
from .functions import (
    aggregate_close_points,
    dimension_slicer,
    norm_img,
    rings_detection,
)

RING_DIAMETER = 0.6  # microns
MAX_ACCUMULATOR_BYTES = 128 * 1024**2


def _average_ring_crops(centers, process, max_accumulator_bytes=MAX_ACCUMULATOR_BYTES):
    """Average the most common crop shape, bounding shape-accumulator memory."""
    groups = {}
    accumulator_bytes = 0
    overflow = False
    for y, x in centers:
        result = process(y, x)
        if result is None:
            continue
        coords, ring = result
        shape = ring.shape
        if shape not in groups:
            size = ring.size * np.dtype(np.float64).itemsize
            if groups and accumulator_bytes + size > max_accumulator_bytes:
                overflow = True
                break
            groups[shape] = {"sum": np.zeros(shape, dtype=np.float64), "coords": []}
            accumulator_bytes += size
        groups[shape]["sum"] += ring
        groups[shape]["coords"].append(coords)

    if overflow:
        # Unusually many crop shapes: use two additional passes rather than
        # retaining an unbounded number of full-volume accumulators.
        groups.clear()
        shapes = {}
        for y, x in centers:
            result = process(y, x)
            if result is not None:
                coords, ring = result
                shapes.setdefault(ring.shape, []).append((y, x, coords))
        if not shapes:
            return None
        shape = max(shapes, key=lambda key: len(shapes[key]))
        total = np.zeros(shape, dtype=np.float64)
        coords_list = []
        for y, x, coords in shapes[shape]:
            result = process(y, x)
            if result is not None and result[1].shape == shape:
                total += result[1]
                coords_list.append(coords)
        if not coords_list:
            return None
        return np.asarray(coords_list), total / len(coords_list)

    if not groups:
        return None
    shape = max(groups, key=lambda key: len(groups[key]["coords"]))
    selected = groups[shape]
    coords = np.asarray(selected["coords"])
    return coords, selected["sum"] / len(coords)


def process_ring_crop(
    img: np.ndarray,
    u_f: float,
    crop_size: int,
) -> np.ndarray | None:
    """
    Process a single ring: upscale, normalize, center and recrop.

    Parameters
    ----------
    img : np.ndarray
        3D ring volume, shape (Z, Y, X).
    u_f : int or float
        Upsampling factor for all axes.
    crop_size : int
        Estimated ring radius (or related) in pixels at original scale.

    Returns
    -------
    np.ndarray or None
        Upsampled and centered ring volume, or None if centering fails.
    """
    if img.ndim != 3:
        raise ValueError(
            f"process_ring_crop expects 3D (Z, Y, X), got shape {img.shape}"
        )

    # Nearest-neighbor zoom can be represented by indices. Locate the crop on
    # the upsampled middle plane first, without allocating the full 3D zoom.
    if not np.isfinite(u_f) or u_f <= 0:
        raise ValueError("Upsampling factor must be positive and finite")
    output_shape = tuple(int(round(size * u_f)) for size in img.shape)
    if min(output_shape) == 0:
        return None
    indices = [
        np.floor(np.linspace(0, size - 1, output_size) + 0.5).astype(np.intp)
        for size, output_size in zip(img.shape, output_shape, strict=True)
    ]
    # For downsampling, zoom does not necessarily include the input extrema.
    extrema_source = img if u_f >= 1 else img[np.ix_(*indices)]
    minimum, maximum = float(np.min(extrema_source)), float(np.max(extrema_source))
    del extrema_source
    if maximum == minimum:
        return None
    mid_slice = img[indices[0][output_shape[0] // 2]][np.ix_(indices[1], indices[2])]
    mid_slice = (mid_slice.astype(float) - minimum) / (maximum - minimum)

    try:
        thresh = filters.threshold_otsu(mid_slice)
    except ValueError:
        return None

    binary = mid_slice > thresh
    labeled_img = measure.label(binary)
    regions = measure.regionprops(labeled_img, intensity_image=mid_slice)
    if not regions:
        return None

    # largest by area
    region = max(regions, key=lambda r: r.area)

    # robust centroid: prefer weighted_centroid if available
    cy, cx = getattr(region, "weighted_centroid", region.centroid)

    # cropping window in upsampled pixels
    f = 0.2  # correction factor
    half_size = int(crop_size * u_f * f)

    H, W = mid_slice.shape
    y1 = max(0, int(cy) - half_size)
    y2 = min(H, int(cy) + half_size)
    x1 = max(0, int(cx) - half_size)
    x2 = min(W, int(cx) + half_size)

    if (y2 <= y1) or (x2 <= x1):
        return None

    ring = img[np.ix_(indices[0], indices[1][y1:y2], indices[2][x1:x2])]
    ring = (ring.astype(float) - minimum) / (maximum - minimum)
    ring = filters.median(ring)

    return ring


def error_return() -> dict[str, Any]:
    """
    Return a standardized error dictionary for RSF analysis.

    Returns
    -------
    dict[str, Any]
        Dictionary indicating an error in RSF analysis.
    """
    return {
        "figure": None,
        "figure name": "Ring_Spread_Function",
        "status": "error",
        "measurements": [
            {"category": "RSF", "name": "FWHM XY", "value": None, "unit": "µm"},
            {"category": "RSF", "name": "FWHM Z", "value": None, "unit": "µm"},
            {
                "category": "RSF",
                "name": "Michelson Contrast",
                "value": None,
                "unit": "",
            },
        ],
        "extra_measurements": {},
    }


def RSF_analysis(
    img_array: np.ndarray,
    metadata: dict,
    visu: bool = True,
    debug: bool = False,
    upsampling_factor: int = 5,
    extracrop_f: int = 8,
    dim="ZYX",
) -> dict[str, Any]:
    """
    Perform the ring spread function analysis on 2D rings using the Argolight 2D ring pattern.

    Parameters
    ----------
    img_array : np.ndarray
        The whole 21x21 ring grid, assumed shape (Z, Y, X).
        Z should be oversampled if possible (e.g. 0.2 µm steps).
    metadata : dict
        Must contain:
            - 'Physical pixel size X' (µm)
            - 'Physical pixel size Z' (µm)
    visu : bool
        Whether to generate the visualization figure.
    debug : bool
        Enable verbose logging.
    upsampling_factor : int
        Upsampling factor for ring extraction.
    extracrop_f : int
        Multiplicative factor for initial crop radius.

    Returns
    -------
    Dict[str, Any]
        Dictionary with figure, status, measurements and extras.
    """

    pxl_size_xy = float(metadata.get("Physical pixel size X", 1.0))
    pxl_size_z = float(metadata.get("Physical pixel size Z", 1.0))

    if (
        int(metadata.get("Image Size C", 0)) > 1
    ):  # in case of multi channel, take the first one
        img_array = img_array[:, :1, ...]
        if debug:
            logger.debug(
                f"[RSF_analysis]: taking first channel for analysis: shape now {img_array.shape}"
            )
    if (
        int(metadata.get("Image Size T", 0)) > 1
    ):  # in case of multi time points, take the first one
        img_array = img_array[:1, ...]
        if debug:
            logger.debug(
                f"[RSF_analysis]: taking first time point for analysis: shape now {img_array.shape}"
            )
    img_array = dimension_slicer(img_array, metadata, dim)

    if img_array.ndim != 3:
        raise ValueError(
            f"[RSF_analysis] expects a 3D stack (Z, Y, X), got shape {img_array.shape}"
        )

    radius_px = max(int(RING_DIAMETER / pxl_size_xy), 2)
    if debug:
        logger.debug(f"[RSF_analysis]: Ring expected radius is {radius_px} pixel.")

    mid = img_array.shape[0] // 2
    # middle plane as reference for ring positions (assume best focus)
    # Detect in the original camera frame: these coordinates are also used to
    # extract volumes from img_array and to draw the detection overlay.
    img_array_mid = norm_img(img_array[mid])

    # detect rings
    binary = rings_detection(img_array_mid, metadata)
    labeled_img = measure.label(binary)
    regions = measure.regionprops(labeled_img, intensity_image=img_array_mid)

    if not regions:
        if debug:
            logger.debug("[RSF_analysis]: no rings detected on middle plane.")
        return error_return()

    # filter by the average size of the detected dots
    area_r = [x.area for x in regions]
    m = np.mean(area_r)
    s = np.std(area_r)
    regions = [x for x in regions if x.area > m - s and x.area < m + s]
    if len(regions) == 0:
        logger.debug("[RSF_analysis]: no rings remaining after area filtering.")
        return error_return()

    # ring centers on middle plane (Y, X)
    coo = []
    for r in regions:
        y, x = getattr(r, "weighted_centroid", r.centroid)
        # Morphological closing can create a region whose original intensities
        # sum to zero, leaving its intensity-weighted centroid undefined.
        if not np.isfinite((y, x)).all():
            y, x = r.centroid
        coo.append((y * pxl_size_xy, x * pxl_size_xy))
    coo = np.array(coo)

    # aggregate close points
    coo, _ = aggregate_close_points(coo, diameter_um=RING_DIAMETER * 5, twoD=True)

    # retranslate them in pixel coordinates
    coo = np.array([[y / pxl_size_xy, x / pxl_size_xy] for y, x in coo])
    if debug:
        logger.debug(f"[RSF_analysis]: Found {len(coo)} rings on the middle plane")

    crop_size = int(radius_px * extracrop_f)  # radius, in pxl

    # Process one ring at a time to bound temporary memory even on machines with
    # many CPUs. Accumulate by output shape to retain the previous modal-shape
    # selection without storing every upsampled volume.
    def _process_single_ring(y: float, x: float):
        y = int(round(y))
        x = int(round(x))

        if (
            y - crop_size < 0
            or y + crop_size > img_array.shape[1]
            or x - crop_size < 0
            or x + crop_size > img_array.shape[2]
        ):
            return None

        ring_vol = img_array[
            :,
            y - crop_size : y + crop_size,
            x - crop_size : x + crop_size,
        ]

        ring_up = process_ring_crop(ring_vol, upsampling_factor, crop_size)
        if ring_up is None:
            return None
        return (y, x), ring_up

    averaged = _average_ring_crops(coo, _process_single_ring)
    if averaged is None:
        if debug:
            logger.debug("[RSF_analysis]: no ring volumes successfully processed.")
        return error_return()

    rings_coo, rings = averaged
    most_common_shape = rings.shape
    if debug:
        logger.debug(f"[RSF_analysis]: Most common shape: {most_common_shape}")

    rings_n = len(rings_coo)

    if debug:
        logger.debug(f"[RSF_analysis]: Rings analyzed: {rings_n}")

    # average ring volume
    rings = filters.gaussian(rings)
    rings = norm_img(rings)

    cz, cy, cx = rings.shape[0] // 2, rings.shape[1] // 2, rings.shape[2] // 2

    # XY analysis
    radius = int(min(cy, cx))
    polar = transform.warp_polar(rings[cz], center=(cy, cx), radius=radius)
    radial_profile = np.mean(polar, axis=0)
    radial_profile = ndimage.uniform_filter1d(radial_profile, size=5)

    baseline = np.mean(radial_profile[:2])
    peak = np.max(radial_profile)
    half_level = baseline + (peak - baseline) / 2
    contrast = (peak - baseline) / (peak + baseline)
    if debug and contrast is not None:
        logger.debug(
            f"[RSF_analysis]: Contrast (Michelson) center/peak is {round(contrast, 3)}."
        )

    fwhm_um = None

    if contrast is not None and (peak - baseline) > 0.02:
        above = radial_profile > half_level
        idx = np.where(np.diff(above.astype(int)) != 0)[0]

        if len(idx) >= 2:
            fwhm_px = idx[-1] - idx[0]
            fwhm_um = fwhm_px * pxl_size_xy / upsampling_factor

        else:  # Fallback, treat like a blob instead
            half_level_blob = (np.min(radial_profile) + peak) / 2
            above_blob = radial_profile > half_level_blob
            idx_blob = np.where(np.diff(above_blob.astype(int)) != 0)[0]
            if len(idx_blob) > 0:
                fwhm_px = idx_blob[-1] * 2  # assume we got half the blob
                fwhm_um = fwhm_px * pxl_size_xy / upsampling_factor

    # Z analysis
    selection_array = np.where(rings[cz] > filters.threshold_otsu(rings[cz]), 1, np.nan)
    z_profile = np.nanmean(rings * selection_array, axis=(1, 2))
    z_profile = ndimage.uniform_filter1d(z_profile, size=5)

    half_level_z = np.min(z_profile) + (np.max(z_profile) - np.min(z_profile)) / 2
    above_z = z_profile > half_level_z
    idx_z = np.where(np.diff(above_z.astype(int)) != 0)[0]

    fwhm_z_um = None

    if len(idx_z) >= 2:
        fwhm_px = idx_z[-1] - idx_z[0]
        fwhm_z_um = fwhm_px * pxl_size_z / upsampling_factor
    else:  # not possible to resolve the ring in Z
        half_level_z_blob = (np.min(z_profile) + peak) / 2
        above_z_blob = z_profile > half_level_z_blob
        idx_z_blob = np.where(np.diff(above_z_blob.astype(int)) != 0)[0]
        if len(idx_z_blob) > 0:
            fwhm_px = idx_z_blob[-1] * 2  # assume we got half the blob
            fwhm_z_um = fwhm_px * pxl_size_z / upsampling_factor

    # visualization
    if visu:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        ax1.imshow(img_array_mid)
        ax1.scatter(rings_coo[:, 1], rings_coo[:, 0], c="r", s=1.5, alpha=0.6)
        ax1.set_title(f"{rings_n} detected rings")
        ax1.axis("off")

        ax2.imshow(rings[cz], origin="lower")
        ax2.scatter(cy, cx, c="r")
        ax2.plot([0, rings.shape[1]], [cy, cy], "k--")
        ax2.plot([cx, cx], [rings.shape[2], 0], "r--")
        ax2.set_title("Average ring")
        ax2.axis("off")

        half = pxl_size_xy / upsampling_factor * len(rings[cz, cy, :]) / 2
        ax3.plot(
            np.arange(len(radial_profile)) * pxl_size_xy / upsampling_factor + half,
            radial_profile,
            "g",
            label="Radial profile",
        )
        ax3.plot(
            np.arange(len(rings[cz, cy, :])) * pxl_size_xy / upsampling_factor,
            rings[cz, cy, :],
            "k--",
            label="Y-profile",
        )
        ax3.plot(
            np.arange(len(rings[cz, :, cx])) * pxl_size_xy / upsampling_factor,
            rings[cz, :, cx],
            "r--",
            label="X-profile",
        )
        ax3.plot(
            np.arange(len(rings[:, cy, cx])) * pxl_size_z / upsampling_factor,
            rings[:, cy, cx],
            "c--",
            label="Z-profile",
        )
        if contrast is not None and fwhm_um is not None:
            ax3.axhline(
                y=half_level, color="blue", linestyle=":", label="Half max (XY)"
            )
        ax3.grid(True)
        ax3.set_title(
            f"Ring profile with FWHM XY: {round(fwhm_um, 3)}\n and Z: {round(fwhm_z_um, 3)} microns"  # type: ignore
            f" and contrast {round(contrast, 3)}."
        )
        ax3.set_xlabel("Distance (μm)")
        ax3.set_ylabel("Intensity (a.u.")
        ax3.legend()

        plt.tight_layout()

        if __name__ == "__main__":
            plt.show()

    else:
        fig = None

    valid = (
        fwhm_um is not None
        and fwhm_um > 0
        and fwhm_z_um is not None
        and fwhm_z_um > 0
        and contrast is not None
        and contrast > 0
    )
    return {
        "figure": fig if visu else None,
        "figure name": "Ring_Spread_Function",
        "status": "success" if valid else "error",
        "measurements": [
            {"category": "RSF", "name": "FWHM XY", "value": fwhm_um, "unit": "µm"},
            {"category": "RSF", "name": "FWHM Z", "value": fwhm_z_um, "unit": "µm"},
            {
                "category": "RSF",
                "name": "Michelson Contrast",
                "value": contrast,
                "unit": "",
            },
        ],
        "extra_measurements": {
            "half_level_z": half_level_z,
            "rings_n": rings_n,
        },
    }
