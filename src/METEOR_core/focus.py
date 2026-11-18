"""Metric-independent reference-plane selection for metrology images."""

import numpy as np
from scipy import ndimage


def _stack_percentiles(stack):
    """Use an exact histogram for small integer types; bound float sampling."""
    if np.issubdtype(stack.dtype, np.integer) and stack.dtype.itemsize <= 2:
        limits = np.iinfo(stack.dtype)
        counts = np.zeros(int(limits.max) - int(limits.min) + 1, dtype=np.int64)
        for plane in stack:
            pixels = np.asarray(plane, dtype=np.int32).ravel() - int(limits.min)
            counts += np.bincount(pixels, minlength=len(counts))
        cumulative = np.cumsum(counts)
        positions = (np.asarray([1.0, 99.9]) / 100) * (stack.size - 1)
        lower = np.floor(positions).astype(np.int64)
        upper = np.ceil(positions).astype(np.int64)
        lo = np.searchsorted(cumulative, lower, side="right") + int(limits.min)
        hi = np.searchsorted(cumulative, upper, side="right") + int(limits.min)
        return lo + (hi - lo) * (positions - lower)

    # For large float stacks, sample evenly across all planes to avoid a
    # full-stack copy. Small stacks retain the exact NumPy percentile behavior.
    per_plane = max(1, 1_000_000 // len(stack))
    values = []
    for plane in stack:
        if plane.size > per_plane:
            indices = np.linspace(0, plane.size - 1, per_plane, dtype=np.intp)
            sample = plane.flat[indices]
        else:
            sample = plane.ravel()
        values.append(sample[np.isfinite(sample)])
    values = np.concatenate(values)
    if not values.size:
        raise ValueError("No finite pixels in focus stack")
    return np.percentile(values, [1, 99.9])


def select_focus_plane(stack: np.ndarray) -> tuple[int, list[float | None]]:
    """Select the plane with the strongest smoothed gradient energy.

    The same robust intensity scale is used for the entire Z stack. Gaussian
    smoothing reduces sensitivity to pixel noise; the score is still a focus
    proxy and is not a guarantee of correct focus for every target or image.
    Integer stacks up to 16 bits use exact percentiles. Large floating-point
    stacks use a deterministic sample of up to approximately one million
    pixels, so their percentile estimates and selected plane can differ from
    a full-stack calculation. Ties select the first plane. Blank and non-finite
    planes are invalid.
    """
    stack = np.asarray(stack)
    if stack.ndim != 3 or stack.shape[0] == 0 or min(stack.shape[1:]) < 2:
        raise ValueError("Expected a non-empty ZYX stack with 2D planes")
    low, high = _stack_percentiles(stack)
    if high <= low:
        raise ValueError("Focus stack has no measurable contrast")
    scores = []
    for plane in stack:
        if not np.isfinite(plane).all():
            scores.append(None)
            continue
        scaled = np.clip((plane.astype(float) - low) / (high - low), 0, 1)
        smooth = ndimage.gaussian_filter(scaled, sigma=1.0)
        dy, dx = np.gradient(smooth)
        score = float(np.mean(dx * dx + dy * dy))
        scores.append(score if score > 0 and np.isfinite(score) else None)
    if all(score is None for score in scores):
        raise ValueError("No usable focus plane")
    best = max(
        range(len(scores)), key=lambda i: scores[i] if scores[i] is not None else -1
    )
    return best, scores
