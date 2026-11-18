import METEOR_core.ring_lattice as ring_lattice
import numpy as np
import pytest
from METEOR_core import coregistration_analysis, uniformity_analysis
from METEOR_core.ring_lattice import match_points, register_ring_lattice


def test_registration_rotation_missing_outliers_and_translation():
    rng = np.random.default_rng(42)
    x, y = np.meshgrid(np.arange(9), np.arange(9))
    grid = np.column_stack((x.ravel(), y.ravel()))
    angle = np.deg2rad(17)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    points = grid @ rotation.T * 20 + [80, 50]
    points += rng.normal(0, 0.08, points.shape)
    points = np.delete(points, [3, 15, 31, 50], axis=0)
    points = np.vstack([points, points[2] + [1, 0], [13, 27]])
    result = register_ring_lattice(points, 20, grid_size=9)
    assert result.diagnostics["matched_count"] == 77
    assert result.diagnostics["unmatched_detections"] == 2
    assert result.diagnostics["residual_p95_px"] < 0.3
    assert len(np.unique(result.detection_indices)) == 77
    shifted = register_ring_lattice(points + [13, -7], 20, grid_size=9)
    assert shifted.diagnostics["matched_count"] == 77
    assert np.allclose(
        shifted.diagnostics["residual_p95_px"],
        result.diagnostics["residual_p95_px"],
        atol=1e-4,
    )


def test_partial_grid_and_duplicate_assignment():
    x, y = np.meshgrid(np.arange(5), np.arange(4))
    points = np.column_stack((x.ravel(), y.ravel())) * 12 + [5, 7]
    result = register_ring_lattice(points, 12, grid_size=21)
    assert result.diagnostics["matched_count"] == 20
    assert result.diagnostics["index_extent"] == [5, 4]
    pairs = match_points([[0, 0]], [[0.1, 0], [0.2, 0]], 1)
    assert pairs.tolist() == [[0, 0]]


def test_invalid_or_ambiguous_grid_is_rejected():
    with pytest.raises(ValueError):
        register_ring_lattice(np.zeros((9, 2)), 10)


def test_detection_selects_threshold_with_most_matches(monkeypatch):
    x, y = np.meshgrid(np.arange(5), np.arange(5))
    points = np.column_stack((x.ravel(), y.ravel())) * 10 + 15

    def detect(image, count, min_size, thr):
        return points[:20] if thr == "triangle" else points

    monkeypatch.setattr(ring_lattice, "get_centroid", detect)
    result = ring_lattice.detect_ring_lattice(np.zeros((80, 80)), 10, grid_size=5)
    assert result.diagnostics["detection_threshold"] == "otsu"
    assert result.diagnostics["matched_count"] == 25


def test_coregistration_uses_observed_offsets(monkeypatch):
    x, y = np.meshgrid(np.arange(5), np.arange(5))
    blue = np.column_stack((x.ravel(), y.ravel())) * 10 + 15
    red = blue + [2, 1]
    red[12] += [1, 0]
    registration = register_ring_lattice(blue, 10, grid_size=5)
    registration.diagnostics["detection_threshold"] = "triangle"
    monkeypatch.setattr(
        coregistration_analysis, "detect_ring_lattice", lambda *a, **kw: registration
    )
    monkeypatch.setattr(coregistration_analysis, "get_centroid", lambda *a, **kw: red)
    image = np.zeros((2, 80, 80))
    result = coregistration_analysis.get_coregistration(
        image, 0.5, spacing=5, grid_size=5
    )
    offsets = result["red_points"] - result["blue_points"]
    assert np.allclose(np.median(offsets, axis=0), [2, 1])
    assert np.max(offsets[:, 0]) == 3
    assert result["channel_matches"] == 25


def test_uniformity_does_not_treat_clipped_rois_as_dark(monkeypatch):
    x, y = np.meshgrid(np.arange(4), np.arange(4))
    points = np.column_stack((x.ravel(), y.ravel())) * 10 + [1, 10]
    registration = register_ring_lattice(points, 10, grid_size=4)
    monkeypatch.setattr(
        uniformity_analysis, "detect_ring_lattice", lambda *a, **kw: registration
    )
    image = np.tile(np.arange(60, dtype=float), (60, 1))
    result = uniformity_analysis.get_uniformity(image, grid_size=4, pxl_size_xy=0.5)
    assert result is not None
    assert result["clipped_roi_count"] == 4
    assert result["valid_roi_count"] == 12
    assert np.isnan(result["final_image"]).sum() == 4


def test_uniformity_excludes_central_cross(monkeypatch):
    x, y = np.meshgrid(np.arange(5), np.arange(5))
    points = np.column_stack((x.ravel(), y.ravel())) * 10 + 10
    registration = register_ring_lattice(points, 10, grid_size=5)
    monkeypatch.setattr(
        uniformity_analysis, "detect_ring_lattice", lambda *a, **kw: registration
    )
    image = np.zeros((70, 70))
    for x, y in points:
        image[y - 2 : y + 3, x - 2 : x + 3] = 10
    image[28:33, 28:33] = 1000
    result = uniformity_analysis.get_uniformity(image, grid_size=5, pxl_size_xy=0.5)
    assert result["excluded_center"]
    assert np.isnan(result["final_image"][2, 2])
    assert result["valid_roi_count"] == 24
    assert result["uniformity"] > 99
