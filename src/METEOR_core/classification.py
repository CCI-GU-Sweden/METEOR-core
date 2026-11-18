"""Conservative, explainable image classification for metrology routing.

Scores are heuristic evidence strengths, not calibrated probabilities. Only
specific pattern detectors can authorize automatic routing; a filename is a hint.
"""

import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
from skimage import filters, measure, morphology, transform

from .focus import select_focus_plane
from .functions import norm_img
from .intensity_analysis import _detect_pattern
from .ring_lattice import detect_ring_lattice

RINGS = "Argo_2D_Rings_Analysis"
INTENSITY = "Argo_Intensity_Linearity"
PSF = "Psf_Measure"
PLASTIC = "Plastic_Field_Uniformity"
STAIRS = "3D_Crossing_Stairs"
STAR = "star_resolution"
CATEGORIES = (RINGS, INTENSITY, PSF, PLASTIC, STAIRS, STAR)


def filename_hint(name):
    """Return a weak filename hint, or Unknown for absent/ambiguous names."""
    name = str(name or "").lower()
    hits = []
    for category, matches in (
        (RINGS, "ring" in name and "2d" in name),
        (INTENSITY, "intensity" in name),
        (PSF, "psf" in name or "bead" in name),
        (PLASTIC, "plastic" in name or "platic" in name),
        (STAIRS, "cross" in name or "stair" in name),
        (STAR, "star" in name),
    ):
        if matches:
            hits.append(category)
    return hits[0] if len(hits) == 1 else "Unknown"


def _canonical_category(value):
    if not isinstance(value, str):
        return None
    aliases = {
        "2d rings": RINGS,
        "rings": RINGS,
        "intensity": INTENSITY,
        "4x4 intensity": INTENSITY,
        "psf": PSF,
        "beads": PSF,
        "plastic": PLASTIC,
        "crossing stairs": STAIRS,
        "star": STAR,
    }
    text = value.strip().casefold()
    return next((c for c in CATEGORIES if c.casefold() == text), aliases.get(text))


def _reference_plane(image, metadata):
    image = np.asarray(image)
    axes = str(metadata.get("axis_order", "")).upper()
    if image.ndim == 2 and not axes:
        axes = "YX"
    if (
        len(axes) != image.ndim
        or len(set(axes)) != len(axes)
        or not set(axes) <= set("STCZYX")
    ):
        raise ValueError("Image axes are missing or do not match the array")
    if not all(axis in axes for axis in "YX") or axes.index("Y") > axes.index("X"):
        raise ValueError("Expected Y and X image axes in that order")
    if "S" in axes and image.shape[axes.index("S")] > 1:
        raise ValueError("Multiple scenes require explicit scene selection")
    slices = tuple(slice(None) if axis in "ZYX" else 0 for axis in axes)
    stack = image[slices]
    remaining = "".join(axis for axis in axes if axis in "ZYX")
    if remaining == "YX":
        stack = stack[None, ...]
    elif remaining != "ZYX":
        raise ValueError("Expected ZYX image axes")
    selected, scores = select_focus_plane(stack)
    return stack[selected], selected, scores


def _star_evidence(image):
    """Detect a large connected multi-spoke structure; reject weak evidence."""
    image = transform.resize(
        norm_img(image), (256, 256), preserve_range=True, anti_aliasing=True
    )
    mask = image > filters.threshold_otsu(image)
    labels = measure.label(mask)
    regions = measure.regionprops(labels)
    if not regions or not mask.any():
        return None
    region = max(regions, key=lambda r: r.area)
    fraction = region.area / np.count_nonzero(mask)
    y1, x1, y2, x2 = region.bbox
    skeleton = morphology.skeletonize(labels == region.label)
    neighbors = (
        ndimage.convolve(skeleton.astype(int), np.ones((3, 3), int), mode="constant")
        - skeleton
    )
    branches = int(np.count_nonzero(skeleton & (neighbors >= 3)))
    # A single branchy component is necessary but not proof of a star. Inspect
    # radial distribution below to distinguish arbitrary connected shapes.
    cy, cx = region.centroid
    yy, xx = np.nonzero(skeleton)
    radius = np.hypot(yy - cy, xx - cx)
    outer = radius > 0.25 * min(y2 - y1, x2 - x1)
    angles = np.arctan2(yy[outer] - cy, xx[outer] - cx)
    bins = np.histogram(angles, bins=72, range=(-np.pi, np.pi))[0]
    peaks, _ = find_peaks(
        np.r_[bins[-2:], bins, bins[:2]],
        prominence=max(2, bins.max() * 0.18),
        distance=3,
    )
    radial_peaks = int(np.count_nonzero((peaks >= 2) & (peaks < 74)))
    if (
        fraction >= 0.8
        and min(y2 - y1, x2 - x1) >= 64
        and branches >= 5
        and radial_peaks >= 8
    ):
        return {
            "foreground_fraction_in_largest_component": float(fraction),
            "radial_peaks": radial_peaks,
            "skeleton_branches": branches,
        }
    return None


def _sparse_bead_evidence(image, z_count):
    if z_count < 3:
        return None
    scaled = transform.resize(
        norm_img(image), (256, 256), preserve_range=True, anti_aliasing=True
    )
    mask = scaled > filters.threshold_otsu(scaled)
    regions = measure.regionprops(measure.label(mask))
    if not regions or not mask.any():
        return None
    small = sum(1 for r in regions if 1 <= r.area <= 20)
    if small >= 30 and small / len(regions) >= 0.8 and mask.mean() < 0.04:
        return {"small_components": small, "foreground_fraction": float(mask.mean())}
    return None


def classify_image(image, metadata, name=None, *, override=None):
    """Return category, evidence and review status; no confidence probabilities.

    Explicit override wins. Metadata may carry an exact `pattern_type` or
    `analysis_type` annotation. Filename hints never authorize routing alone.
    Only first channel/time point is inspected; multiple scenes are rejected.
    """
    metadata = metadata or {}
    result = {
        "category": "Unknown",
        "confidence": None,
        "source": None,
        "candidates": [],
        "evidence": [],
        "warnings": [],
        "requires_review": True,
    }
    if override is not None:
        category = _canonical_category(override)
        if category is None:
            raise ValueError(f"Unsupported classification override: {override}")
        result.update(
            category=category,
            confidence="explicit",
            source="override",
            requires_review=False,
        )
        return result
    annotations = []
    for key in ("pattern_type", "analysis_type"):
        if key in metadata and metadata[key] is not None:
            category = _canonical_category(metadata[key])
            if category is None:
                result["warnings"].append(f"Unrecognized {key} annotation")
                return result
            annotations.append((key, category))
    if len({category for _, category in annotations}) > 1:
        result["warnings"].append("Explicit metadata annotations conflict")
        return result
    hint = filename_hint(name)
    if hint != "Unknown":
        result["evidence"].append(
            {"source": "filename", "category": hint, "strength": "weak"}
        )
    try:
        plane, z, _ = _reference_plane(image, metadata)
        size = max(plane.shape)
        factor = min(1.0, 768 / size)
        if factor < 1:
            plane = transform.resize(
                plane,
                (
                    max(2, round(plane.shape[0] * factor)),
                    max(2, round(plane.shape[1] * factor)),
                ),
                preserve_range=True,
                anti_aliasing=True,
            )
        try:
            pixel = float(metadata.get("Physical pixel size X", 0))
        except (TypeError, ValueError):
            pixel = 0
        pixel = pixel / factor if np.isfinite(pixel) and pixel > 0 else None
        result["evidence"].append(
            {"source": "image", "reference_z": z, "reference_channel": 0}
        )
        if pixel is not None:
            try:
                ring = detect_ring_lattice(norm_img(plane), 5.0 / pixel, 21, min_size=1)
                d = ring.diagnostics
                if (
                    d["matched_count"] >= 50
                    and d["matched_count"] / d["detected_count"] >= 0.65
                    and min(d["index_extent"]) >= 7
                    and d["residual_p95_px"] <= 0.2 * d["spacing_px"]
                ):
                    result["candidates"].append(
                        {"category": RINGS, "strength": "strong", "evidence": d}
                    )
            except (ValueError, IndexError, RuntimeError):
                pass
            try:
                _, pattern = _detect_pattern(plane, pixel, 4, 6.0, 10.0)
                d = pattern.diagnostics
                if d["residual_p95_px"] <= 0.15 * d["spacing_px"]:
                    result["candidates"].append(
                        {"category": INTENSITY, "strength": "strong", "evidence": d}
                    )
            except (ValueError, IndexError, RuntimeError):
                pass
        else:
            result["warnings"].append(
                "Pixel size unavailable; physical lattice detectors were skipped"
            )
        star = _star_evidence(plane)
        if star:
            result["candidates"].append(
                {"category": STAR, "strength": "strong", "evidence": star}
            )
        axes = str(metadata.get("axis_order", "")).upper()
        z_count = image.shape[axes.index("Z")] if "Z" in axes else 1
        beads = _sparse_bead_evidence(plane, z_count)
        if beads:
            result["candidates"].append(
                {"category": PSF, "strength": "suggestive", "evidence": beads}
            )
    except (ValueError, TypeError, IndexError, KeyError) as exc:
        result["warnings"].append(f"Image could not be classified: {exc}")
        # A structured annotation is trusted even when image inspection fails,
        # except for multi-scene inputs which require explicit selection.
        if "Multiple scenes" in str(exc):
            return result
    strong = [
        item["category"]
        for item in result["candidates"]
        if item["strength"] == "strong"
    ]
    if annotations:
        category = annotations[0][1]
        if strong and category not in strong:
            result["warnings"].append(
                "Structured metadata conflicts with image evidence"
            )
            return result
        if len(strong) > 1:
            result["warnings"].append("Multiple image patterns are plausible")
            return result
        result.update(
            category=category,
            confidence="explicit",
            source="metadata",
            requires_review=False,
        )
    elif len(strong) == 1:
        result.update(
            category=strong[0],
            confidence="strong",
            source="image",
            requires_review=False,
        )
    elif len(strong) > 1:
        result["warnings"].append("Multiple image patterns are plausible")
    else:
        result["warnings"].append(
            "No sufficiently specific image pattern; select an analysis explicitly"
        )
    if result["category"] != "Unknown" and hint not in ("Unknown", result["category"]):
        result["warnings"].append("Filename hint conflicts with the selected category")
    return result
