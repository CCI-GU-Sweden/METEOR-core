"""Metric-independent reference-plane selection for metrology images."""

import numpy as np
from scipy import ndimage


def select_focus_plane(stack: np.ndarray) -> tuple[int, list[float | None]]:
    """Select the plane with the strongest smoothed gradient energy.

    The same robust intensity scale is used for the entire Z stack. Gaussian
    smoothing reduces sensitivity to pixel noise; the score is still a focus
    proxy and is not a guarantee of correct focus for every target or image.
    Ties select the first plane. Blank and non-finite planes are invalid.
    """
    stack = np.asarray(stack)
    if stack.ndim != 3 or stack.shape[0] == 0 or min(stack.shape[1:]) < 2:
        raise ValueError("Expected a non-empty ZYX stack with 2D planes")
    finite = np.isfinite(stack)
    if not finite.any():
        raise ValueError("No finite pixels in focus stack")
    values = stack[finite].astype(float)
    low, high = np.percentile(values, [1, 99.9])
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
