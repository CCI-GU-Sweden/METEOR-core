"""Registration of detected ring centers in original image coordinates.

The fitted similarity transform removes translation, rotation and uniform scale;
residuals therefore do not measure global magnification error. Indices are
relative: a periodic pattern alone cannot identify the physical grid origin.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.spatial import cKDTree

from .functions import get_centroid


@dataclass
class RingLattice:
    nodes_xy: np.ndarray
    points_xy: np.ndarray
    indices: np.ndarray
    detection_indices: np.ndarray
    spacing_px: float
    angle_rad: float
    diagnostics: dict


def detect_ring_lattice(image, spacing_px, grid_size=21, min_size=20):
    """Try several segmentation thresholds and rank by geometric fit quality."""
    candidates = []
    for method in ("triangle", "otsu", "yen"):
        try:
            points = get_centroid(image, None, min_size=min_size, thr=method)
            if points is None or len(points) > 4 * grid_size * grid_size:
                continue
            registration = register_ring_lattice(points, spacing_px, grid_size)
        except (ValueError, IndexError, RuntimeError):
            continue
        registration.diagnostics["detection_threshold"] = method
        candidates.append(registration)
    if not candidates:
        raise ValueError("No threshold produced a usable ring lattice")
    return max(candidates, key=lambda r: r.diagnostics["matched_count"])


def match_points(reference_xy, detected_xy, max_distance_px):
    """One-to-one gated assignment; return reference and detection indices."""
    reference_xy = np.asarray(reference_xy, dtype=float).reshape(-1, 2)
    detected_xy = np.asarray(detected_xy, dtype=float).reshape(-1, 2)
    if not np.isfinite(max_distance_px) or max_distance_px <= 0:
        raise ValueError("Matching distance must be positive")
    if not len(reference_xy) or not len(detected_xy):
        return np.empty((0, 2), dtype=int)
    neighbors = cKDTree(reference_xy).sparse_distance_matrix(
        cKDTree(detected_xy), max_distance_px, output_type="coo_matrix"
    )
    # Only gated edges and one private unmatched edge per reference are stored.
    # A positive offset preserves exact zero-distance edges in the sparse graph.
    keep = neighbors.data < max_distance_px
    offset = np.finfo(float).eps * max(1.0, max_distance_px)
    n = len(reference_xy)
    row = np.concatenate((neighbors.row[keep], np.arange(n)))
    col = np.concatenate((neighbors.col[keep], len(detected_xy) + np.arange(n)))
    costs = np.concatenate(
        (neighbors.data[keep] + offset, np.full(n, max_distance_px + offset))
    )
    graph = coo_matrix((costs, (row, col)), shape=(n, len(detected_xy) + n)).tocsr()
    rows, cols = min_weight_full_bipartite_matching(graph)
    real = cols < len(detected_xy)
    return np.column_stack((rows[real], cols[real]))


def _limit_to_grid(pairs, indices, grid_size):
    """Keep the densest grid-sized window of matched integer coordinates."""
    coords = indices[pairs[:, 0]]
    if np.all(np.ptp(coords, axis=0) < grid_size):
        return pairs
    starts_x = np.unique(np.concatenate((coords[:, 0], coords[:, 0] - grid_size + 1)))
    starts_y = np.unique(np.concatenate((coords[:, 1], coords[:, 1] - grid_size + 1)))
    best = np.zeros(len(pairs), dtype=bool)
    for x in starts_x:
        x_inside = (coords[:, 0] >= x) & (coords[:, 0] < x + grid_size)
        for y in starts_y:
            inside = x_inside & (coords[:, 1] >= y) & (coords[:, 1] < y + grid_size)
            if inside.sum() > best.sum():
                best = inside
    return pairs[best]


def register_ring_lattice(points_xy, spacing_px, grid_size=21, gate_fraction=0.35):
    """Fit a square similarity lattice and match observations one-to-one.

    This uses nearest-neighbor vectors to initialize orientation and pitch,
    then iteratively fits robust residuals and gated assignments. The pitch is
    bounded to ±15% of the supplied nominal value. Partial patterns are allowed;
    minimum matched count and two-dimensional extent are required. Indices are
    relative and cannot resolve whole-pitch shifts or 90-degree symmetries.
    """
    points = np.asarray(points_xy, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or len(points) < 9
        or not np.isfinite(points).all()
    ):
        raise ValueError("At least nine finite (x, y) detections are required")
    if not np.isfinite(spacing_px) or spacing_px <= 0 or grid_size < 2:
        raise ValueError("Invalid grid size or nominal spacing")
    if not 0 < gate_fraction < 0.5:
        raise ValueError("Gate fraction must lie between 0 and 0.5")

    pairs = np.array(
        list(cKDTree(points).query_pairs(1.25 * spacing_px)), dtype=int
    ).reshape(-1, 2)
    if not len(pairs):
        raise ValueError("No neighboring ring detections")
    vectors = points[pairs[:, 1]] - points[pairs[:, 0]]
    lengths = np.linalg.norm(vectors, axis=1)
    vectors = vectors[(lengths > 0.75 * spacing_px) & (lengths < 1.25 * spacing_px)]
    if len(vectors) < 6:
        raise ValueError("Insufficient neighboring ring detections")
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    coherence = np.mean(np.exp(4j * angles))
    if abs(coherence) < 0.3:
        raise ValueError("Ring orientation is ambiguous")
    angle = float(np.angle(coherence) / 4)
    pitch = float(
        np.clip(
            np.median(np.linalg.norm(vectors, axis=1)),
            0.85 * spacing_px,
            1.15 * spacing_px,
        )
    )

    def rotation(theta):
        return np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )

    local = points @ rotation(angle)
    phase = (
        np.angle(np.mean(np.exp(2j * np.pi * local / pitch), axis=0))
        * pitch
        / (2 * np.pi)
    )
    origin = phase @ rotation(angle).T
    params = np.array([origin[0], origin[1], angle, pitch])
    matches = None
    for _ in range(6):
        origin, angle, pitch = params[:2], params[2], params[3]
        local = (points - origin) @ rotation(angle) / pitch
        integer = np.rint(local).astype(int)
        # Candidate nodes include each observed nearest node, with a one-node
        # margin so close competing detections can be assigned correctly.
        offsets = np.array([(x, y) for x in (-1, 0, 1) for y in (-1, 0, 1)])
        indices = np.unique((integer[:, None, :] + offsets).reshape(-1, 2), axis=0)
        nodes = indices @ rotation(angle).T * pitch + origin
        pairs = _limit_to_grid(
            match_points(nodes, points, gate_fraction * pitch), indices, grid_size
        )
        if len(pairs) < 9:
            raise ValueError("Insufficient one-to-one lattice matches")
        selected_indices = indices[pairs[:, 0]]
        selected_points = points[pairs[:, 1]]

        def residual(
            p, selected_indices=selected_indices, selected_points=selected_points
        ):
            predicted = selected_indices @ rotation(p[2]).T * p[3] + p[:2]
            return (predicted - selected_points).ravel()

        fit = least_squares(
            residual,
            params,
            bounds=(
                [-np.inf, -np.inf, angle - 0.2, 0.85 * spacing_px],
                [np.inf, np.inf, angle + 0.2, 1.15 * spacing_px],
            ),
            loss="soft_l1",
            f_scale=0.1 * spacing_px,
        )
        if not fit.success:
            raise ValueError("Lattice fit did not converge")
        matches = pairs
        change = np.linalg.norm(fit.x - params)
        params = fit.x
        if change < 1e-5:
            break

    origin, angle, pitch = params[:2], params[2], params[3]
    nodes = indices @ rotation(angle).T * pitch + origin
    matches = _limit_to_grid(
        match_points(nodes, points, gate_fraction * pitch), indices, grid_size
    )
    idx, det = matches[:, 0], matches[:, 1]
    selected_indices = indices[idx]
    extent = np.ptp(selected_indices, axis=0) + 1
    if len(matches) < 9 or np.min(extent) < 3 or np.max(extent) > grid_size:
        raise ValueError("Matched lattice has insufficient or implausible extent")
    residuals = np.linalg.norm(nodes[idx] - points[det], axis=1)
    return RingLattice(
        nodes[idx],
        points[det],
        selected_indices,
        det,
        float(pitch),
        float(angle),
        {
            "matched_count": int(len(matches)),
            "detected_count": int(len(points)),
            "unmatched_detections": int(len(points) - len(matches)),
            "expected_count": int(grid_size * grid_size),
            "index_extent": extent.tolist(),
            "residual_median_px": float(np.median(residuals)),
            "residual_p95_px": float(np.percentile(residuals, 95)),
            "spacing_px": float(pitch),
            "angle_deg": float(np.degrees(angle)),
            "indices_are_relative": True,
            "fit_model": "similarity",
        },
    )
