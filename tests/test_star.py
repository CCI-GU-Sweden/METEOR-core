# tests/test_star.py

import importlib.util

import METEOR_core.functions as f
import METEOR_core.star_resolution as sr
import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_star_resolution_analysis_tiny_2d():
    # Skip if the optional CZI reader dependency isn't installed
    if importlib.util.find_spec("pylibCZIrw") is None:
        import pytest

        pytest.skip("pylibCZIrw not installed; skipping CZI-based integration test")

    img, metadata = f.czi_reader_tzcxy("tests/data/test_star.czi")

    result = sr.star_resolution_analysis(img, metadata, visu=False, debug=False)

    assert isinstance(result, dict)
    assert result["figure"] is None
    assert result["figure name"] == "star_resolution"

    assert result["status"] in {"success", "error"}
    measurements = result.get("measurements")
    assert isinstance(measurements, list)
    assert len(measurements) == 1

    meas = measurements[0]
    assert meas["category"] == "Star resolution"
    assert meas["name"] == "Half-pitch resolution"
    assert meas["unit"] == "µm"

    res_um = meas["value"]

    # For a synthetic star, we expect it to succeed and be finite.
    assert result["status"] == "success"
    assert isinstance(res_um, float)
    assert np.isfinite(res_um)
    assert res_um > 0.0

    # Loose sanity bounds: should be in a plausible microscopy range.
    # (Adjust if your synthetic parameters change.)
    assert 0.05 <= res_um <= 5.0

    extra = result.get("extra_measurements", {})
    assert "resolution_length_um" in extra
    assert extra["resolution_length_um"] == res_um
    assert "radial_profile" in extra
    assert "radial_freqs" in extra
