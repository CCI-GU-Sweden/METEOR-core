import numpy as np
import pytest
from METEOR_core.intensity_analysis import (
    _fit_expected,
    _measure_elements,
    intensity_analysis,
)
from METEOR_core.ring_lattice import register_ring_lattice


def make_pattern():
    image = np.full((80, 80), 20.0)
    xx, yy = np.meshgrid(np.arange(4), np.arange(4))
    centers = np.column_stack((xx.ravel(), yy.ravel())) * 12 + 20
    for index, (x, y) in enumerate(centers):
        image[y - 3 : y + 4, x - 3 : x + 4] += (index + 1) * 10
    return image, centers


def test_raw_local_background_and_spatial_fit():
    image, centers = make_pattern()
    reg = register_ring_lattice(centers, 12, grid_size=4)
    elements, _, _ = _measure_elements(image, reg, 1, 6, 4, None)
    assert [e["signal"] for e in elements] == list(np.arange(1, 17) * 10)
    assert all(e["background"] == 20 for e in elements)
    fit = _fit_expected(elements, np.arange(1, 17).reshape(4, 4), 4)
    assert fit["r2"] == pytest.approx(1)
    assert fit["slope"] == pytest.approx(10)
    wrong_layout = np.arange(1, 17).reshape(4, 4).T
    assert _fit_expected(elements, wrong_layout, 4)["r2"] < 0.5


def test_saturation_and_clipped_background_rejected():
    image, centers = make_pattern()
    reg = register_ring_lattice(centers, 12, grid_size=4)
    with pytest.raises(ValueError, match="saturated"):
        _measure_elements(image, reg, 1, 6, 4, 100)
    reg.nodes_xy -= [20, 0]
    with pytest.raises(ValueError, match="clipped"):
        _measure_elements(image, reg, 1, 6, 4, None)


def test_end_to_end_pattern_and_expected_layout():
    image, _ = make_pattern()
    metadata = {"axis_order": "ZYX", "Physical pixel size X": 1}
    expected = np.arange(1, 17).reshape(4, 4)
    result = intensity_analysis(
        image[np.newaxis], metadata, spacing=12, expected_levels=expected, visu=False
    )
    assert result["status"] == "success", result["extra_measurements"]
    assert result["measurements"][0]["value"] == pytest.approx(1, abs=0.02)
    assert len(result["extra_measurements"]["elements"]) == 16


def test_translation_and_rotation_preserve_camera_frame_assignments():
    image, _ = make_pattern()
    metadata = {"axis_order": "ZYX", "Physical pixel size X": 1}
    expected = np.arange(1, 17).reshape(4, 4)
    shifted = np.roll(image, (3, -4), axis=(0, 1))
    translated = intensity_analysis(
        shifted[np.newaxis], metadata, spacing=12, expected_levels=expected, visu=False
    )
    rotated = intensity_analysis(
        np.rot90(image)[np.newaxis],
        metadata,
        spacing=12,
        expected_levels=np.rot90(expected),
        visu=False,
    )
    assert translated["status"] == "success"
    assert rotated["status"] == "success"
    assert translated["measurements"][0]["value"] == pytest.approx(1, abs=0.02)
    assert rotated["measurements"][0]["value"] == pytest.approx(1, abs=0.02)


def test_missing_calibration_returns_error():
    image, _ = make_pattern()
    result = intensity_analysis(image[np.newaxis], {"axis_order": "ZYX"}, visu=False)
    assert result["status"] == "error"
    assert "Physical pixel size X" in result["extra_measurements"]["error"]
