# tests/test_intensity_analysis.py

import importlib.util

import METEOR_core.functions as f
import METEOR_core.intensity_analysis as ia
import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_intensity_analysis_real_czi():
    # Skip if the optional CZI reader dependency isn't installed
    if importlib.util.find_spec("pylibCZIrw") is None:
        import pytest

        pytest.skip("pylibCZIrw not installed; skipping CZI-based integration test")

    img, metadata = f.czi_reader_tzcxy("tests/data/test_intensity.czi")

    result = ia.intensity_analysis(img, metadata, debug=False)

    assert isinstance(result, dict)
    assert result["status"] in {"success", "error"}  # allow error if data is bad

    measurements = result.get("measurements")
    assert isinstance(measurements, list)
    assert len(measurements) == 1

    meas = measurements[0]
    assert meas["category"] == "Intensity"
    assert meas["name"] == "Linearity of Intensity"

    r2 = meas["value"]
    # If success: r2 should be finite and in [0, 1]
    if result["status"] == "success":
        assert isinstance(r2, float)
        assert np.isfinite(r2)
        assert r2 > 0.90
    else:
        # If error: allow None or NaN
        assert (r2 is None) or (isinstance(r2, float) and not np.isfinite(r2))

    assert "figure" in result
    assert "extra_measurements" in result

    extra = result["extra_measurements"]
    assert "best_z_plane" in extra
    assert (extra["best_z_plane"] is None) or isinstance(extra["best_z_plane"], int)
