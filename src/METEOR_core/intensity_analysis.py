"""Measurement of the Argolight 4×4 intensity-gradation pattern.

The default geometry is intended for Argo-SIM: 6 µm elements on a nominal
10 µm pitch. Verify the slide model and pitch before interpreting results.
Element identities are relative to the image, not the physical slide.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy import ndimage
from skimage.feature import peak_local_max

from . import logger
from .focus import select_focus_plane
from .functions import dimension_slicer, get_label, norm_img
from .ring_lattice import register_ring_lattice


def _detect_pattern(image, pixel_size, grid_size, pattern_size, spacing):
    """Find the complete grid using geometry, not the measured intensities."""
    candidates = []
    normalized = norm_img(image)
    min_distance = max(1, int(round(0.3 * pattern_size / pixel_size)))
    for threshold in ("triangle", "li", "otsu", "yen", "mean"):
        try:
            labels = get_label(normalized, thr=threshold, closing_radius=2)
            distance = ndimage.distance_transform_edt(labels > 0)
            peaks = peak_local_max(
                distance,
                min_distance=min_distance,
                labels=labels > 0,
                exclude_border=True,
            )
            if len(peaks) > 4 * grid_size**2:
                continue
            registration = register_ring_lattice(
                peaks[:, ::-1], spacing / pixel_size, grid_size
            )
            if registration.diagnostics["matched_count"] != grid_size**2:
                continue
            if registration.diagnostics["index_extent"] != [grid_size, grid_size]:
                continue
            if (
                registration.diagnostics["residual_p95_px"]
                > 0.25 * registration.spacing_px
            ):
                continue
            candidates.append(
                (registration.diagnostics["residual_p95_px"], threshold, registration)
            )
        except (ValueError, IndexError, RuntimeError):
            continue
    # A blurred view can merge the substructure of an element when thresholded
    # line patterns fragment. Measurements still use the unblurred raw image.
    smooth = ndimage.gaussian_filter(normalized, sigma=pattern_size / (3 * pixel_size))
    peaks = peak_local_max(
        smooth,
        min_distance=max(1, int(0.4 * spacing / pixel_size)),
        threshold_rel=0.005,
        exclude_border=True,
    )
    if grid_size**2 <= len(peaks) <= 4 * grid_size**2:
        try:
            registration = register_ring_lattice(
                peaks[:, ::-1], spacing / pixel_size, grid_size
            )
            if (
                registration.diagnostics["matched_count"] == grid_size**2
                and registration.diagnostics["index_extent"] == [grid_size, grid_size]
                and registration.diagnostics["residual_p95_px"]
                <= 0.25 * registration.spacing_px
            ):
                candidates.append(
                    (
                        registration.diagnostics["residual_p95_px"],
                        "smoothed_peaks",
                        registration,
                    )
                )
        except (ValueError, IndexError, RuntimeError):
            pass
    if not candidates:
        raise ValueError("Could not identify a complete 4×4 intensity pattern")
    _, threshold, registration = min(candidates, key=lambda item: item[0])
    return threshold, registration


def _measure_elements(
    image,
    registration,
    pixel_size,
    pattern_size,
    grid_size,
    saturation_value,
    max_saturated_fraction=0.05,
):
    """Measure fixed square ROIs and surrounding local background on raw pixels."""
    half = max(1, int(round(pattern_size / pixel_size / 2)))
    outer = int(np.floor(registration.spacing_px * 0.49))
    if outer <= half:
        raise ValueError(
            "Element ROI leaves no local background region; check geometry"
        )
    height, width = image.shape
    elements = []
    relative = registration.indices - registration.indices.min(axis=0)
    for node, (col, row) in zip(registration.nodes_xy, relative, strict=False):
        x, y = (int(round(v)) for v in node)
        if x - outer < 0 or y - outer < 0 or x + outer >= width or y + outer >= height:
            raise ValueError(f"Element ({row}, {col}) or its background is clipped")
        roi = image[y - half : y + half + 1, x - half : x + half + 1].astype(float)
        local = image[y - outer : y + outer + 1, x - outer : x + outer + 1].astype(
            float
        )
        mask = np.ones(local.shape, dtype=bool)
        mask[outer - half : outer + half + 1, outer - half : outer + half + 1] = False
        bg_pixels = local[mask]
        if not np.isfinite(roi).all() or not np.isfinite(bg_pixels).all():
            raise ValueError(f"Element ({row}, {col}) contains non-finite pixels")
        background = float(np.median(bg_pixels))
        bg_noise = float(1.4826 * np.median(np.abs(bg_pixels - background)))
        saturated_fraction = (
            float(np.mean(roi >= saturation_value))
            if saturation_value is not None
            else None
        )
        if (
            saturated_fraction is not None
            and saturated_fraction > max_saturated_fraction
        ):
            raise ValueError(
                f"Element ({row}, {col}) exceeds the allowed saturated pixel fraction "
                f"({saturated_fraction:.1%} > {max_saturated_fraction:.1%})"
            )
        raw_mean = float(np.mean(roi))
        signal = raw_mean - background
        elements.append(
            {
                "row": int(row),
                "column": int(col),
                "center_x_px": float(node[0]),
                "center_y_px": float(node[1]),
                "raw_mean": raw_mean,
                "background": background,
                "background_noise_mad": bg_noise,
                "signal": signal,
                "low_signal": bool(signal <= 2 * bg_noise),
                "saturated": bool(saturated_fraction)
                if saturated_fraction is not None
                else None,
                "saturated_fraction": saturated_fraction,
            }
        )
    elements.sort(key=lambda item: (item["row"], item["column"]))
    return elements, half, outer


def _fit_expected(elements, expected_levels, grid_size):
    """Fit against an explicitly supplied camera-frame response matrix."""
    expected = np.asarray(expected_levels, dtype=float)
    if expected.shape != (grid_size, grid_size) or not np.isfinite(expected).all():
        raise ValueError("expected_levels must be a finite grid_size × grid_size array")
    x = np.asarray([expected[e["row"], e["column"]] for e in elements])
    y = np.asarray([e["signal"] for e in elements])
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        raise ValueError("Expected and observed levels must vary for regression")
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    r2 = 1 - np.sum((y - predicted) ** 2) / np.sum((y - y.mean()) ** 2)
    for e, value, fit in zip(elements, x, predicted, strict=False):
        e["expected_level"] = float(value)
        e["predicted_signal"] = float(fit)
        e["residual"] = float(e["signal"] - fit)
    return {"r2": float(r2), "slope": float(slope), "intercept": float(intercept)}


def _fit_sorted(elements):
    """Descriptive fit of ascending responses to equally spaced ranks."""
    ordered = sorted(elements, key=lambda item: item["signal"])
    y = np.asarray([item["signal"] for item in ordered])
    if np.ptp(y) <= 0:
        raise ValueError("Element responses do not vary; R² cannot be calculated")
    x = np.arange(1, len(y) + 1, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    r2 = 1 - np.sum((y - predicted) ** 2) / np.sum((y - y.mean()) ** 2)
    for rank, element, estimate in zip(x, ordered, predicted, strict=True):
        element["sorted_rank"] = int(rank)
        element["sorted_fit_signal"] = float(estimate)
    return {
        "r2": float(r2),
        "slope": float(slope),
        "intercept": float(intercept),
        "sorted_signals": y.tolist(),
        "predicted": predicted.tolist(),
        "method": "sorted signal versus equally spaced rank",
    }


def _figure(image, elements, half, fit):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(image, cmap="gray")
    for e in elements:
        x, y = e["center_x_px"], e["center_y_px"]
        ax1.add_patch(
            Rectangle(
                (round(x) - half - 0.5, round(y) - half - 0.5),
                2 * half + 1,
                2 * half + 1,
                fill=False,
                edgecolor="red",
                linewidth=0.8,
            )
        )
        ax1.text(
            x, y, f"{e['row']},{e['column']}", color="yellow", fontsize=7, ha="center"
        )
    ax1.set_title("Camera-frame element ROIs")
    ax1.axis("off")
    x = np.arange(1, len(fit["sorted_signals"]) + 1)
    ax2.plot(x, fit["sorted_signals"], "o-", label="Sorted responses")
    ax2.plot(x, fit["predicted"], "r--", label="Linear fit")
    ax2.set_xlabel("Intensity rank")
    ax2.set_title(f"Sorted intensity response: R²={fit['r2']:.3f}")
    ax2.legend()
    ax2.set_ylabel("Background-subtracted signal (raw units)")
    ax2.grid(True)
    fig.tight_layout()
    return fig


def intensity_analysis(
    img_array: np.ndarray,
    metadata: dict,
    grid_size: int = 4,
    pattern_size: float = 6.0,
    spacing: float = 10.0,
    dim="ZYX",
    debug: bool = False,
    stop_value: float = 0.995,
    *,
    expected_levels=None,
    saturation_value: float | None = None,
    max_saturated_fraction: float = 0.05,
    visu: bool = True,
) -> dict[str, Any]:
    """Measure raw 4×4 signals and fit their ascending values against rank.

    `expected_levels` must be an explicitly oriented 4×4 matrix corresponding
    to the reported relative row/column positions. No orientation is inferred
    from intensities. An oriented matrix provides an additional spatial fit.
    `saturation_value` should be the actual detector ceiling;
    dtype limits alone cannot establish a camera's effective bit depth.
    `stop_value` is retained for callers of the previous API and is unused.
    """
    result = {
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
        "extra_measurements": {"best_z_plane": None},
    }
    try:
        pixel_size = float(metadata["Physical pixel size X"])
        pixel_y = float(metadata.get("Physical pixel size Y", pixel_size))
        if (
            not np.isfinite(pixel_size)
            or pixel_size <= 0
            or not np.isclose(pixel_size, pixel_y, rtol=0.01)
        ):
            raise ValueError("Valid, approximately equal X/Y pixel sizes are required")
        if (
            not np.isfinite(pattern_size)
            or not np.isfinite(spacing)
            or pattern_size <= 0
            or spacing <= pattern_size
        ):
            raise ValueError("Invalid pattern width or spacing")
        if int(metadata.get("Image Size C", 1)) > 1:
            img_array = img_array[:, :1, ...]
        if int(metadata.get("Image Size T", 1)) > 1:
            img_array = img_array[:1, ...]
        stack = dimension_slicer(img_array, metadata, dim)
        if stack.ndim == 2:
            stack = stack[np.newaxis]
        plane, scores = select_focus_plane(stack)
        result["extra_measurements"].update(
            {"best_z_plane": plane, "focus_scores": scores}
        )
        image = stack[plane]
        threshold, registration = _detect_pattern(
            image, pixel_size, grid_size, pattern_size, spacing
        )
        result["extra_measurements"].update(
            {"used_threshold": threshold, "registration": registration.diagnostics}
        )
        explicit_ceiling = saturation_value is not None
        if saturation_value is not None and (
            not np.isfinite(saturation_value) or saturation_value <= 0
        ):
            raise ValueError("saturation_value must be finite and positive")
        if saturation_value is None and np.issubdtype(image.dtype, np.integer):
            saturation_value = float(np.iinfo(image.dtype).max)
        if (
            not np.isfinite(max_saturated_fraction)
            or not 0 <= max_saturated_fraction <= 1
        ):
            raise ValueError("max_saturated_fraction must be between 0 and 1")
        elements, half, outer = _measure_elements(
            image,
            registration,
            pixel_size,
            pattern_size,
            grid_size,
            saturation_value,
            max_saturated_fraction,
        )
        spatial_fit = (
            _fit_expected(elements, expected_levels, grid_size)
            if expected_levels is not None
            else None
        )
        fit = _fit_sorted(elements)
        signal = np.asarray([e["signal"] for e in elements])
        signal_range = float(np.max(signal) - np.min(signal))
        warnings = [
            "Physical orientation and absolute element identity are not established."
        ]
        if not explicit_ceiling:
            warnings.append(
                "Detector saturation level is not verified; stored dtype maximum is only an upper bound."
            )
        if any(e["low_signal"] for e in elements):
            warnings.append(
                "At least one element has low signal relative to local background variation."
            )
        if any(e["saturated"] for e in elements):
            warnings.append(
                "Some ROI pixels are saturated; clipping may bias the response and R²."
            )
        warnings.append(
            "Sorted-rank R² assumes equally spaced expected responses and can hide spatial ordering errors; it is not a calibrated camera-linearity measurement."
        )
        if spatial_fit is not None and spatial_fit["slope"] <= 0:
            warnings.append(
                "The fitted response slope is non-positive; verify the expected layout and image orientation."
            )
        result["measurements"][0]["value"] = fit["r2"]
        result["measurements"].append(
            {
                "category": "Intensity",
                "name": "Pattern signal range",
                "value": signal_range,
                "unit": "raw intensity units",
            }
        )
        result["extra_measurements"] = {
            "best_z_plane": plane,
            "focus_scores": scores,
            "focus_at_stack_boundary": len(scores) > 1
            and plane in (0, len(scores) - 1),
            "used_threshold": threshold,
            "registration": registration.diagnostics,
            "elements": elements,
            "fit": fit,
            "spatial_fit": spatial_fit,
            "warnings": warnings,
            "element_coordinates": "relative camera-frame row and column; physical orientation ambiguous",
            "roi_half_size_px": half,
            "background_outer_radius_px": outer,
            "saturation_value": saturation_value,
            "saturation_value_verified": explicit_ceiling,
            "max_saturated_fraction": max_saturated_fraction,
            "pattern_size_um": pattern_size,
            "spacing_um": spacing,
        }
        if visu:
            result["figure"] = _figure(image, elements, half, fit)
        result["status"] = "success"
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        logger.warning(f"Intensity analysis failed: {exc}")
        result["extra_measurements"]["error"] = str(exc)
    return result
