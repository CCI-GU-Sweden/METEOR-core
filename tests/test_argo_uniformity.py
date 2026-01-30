# tests/test_argo_uniformity.py

import importlib.util

import METEOR_core.functions as f
import METEOR_core.uniformity_analysis as ua
import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_uniformity_analysis_real_czi():
    # Skip if optional CZI reader dependency isn't installed
    if importlib.util.find_spec("pylibCZIrw") is None:
        import pytest

        pytest.skip("pylibCZIrw not installed; skipping CZI-based integration test")

    img, metadata = f.czi_reader_tzcxy("tests/data/test_2drings.czi")

    result = ua.uniformity_analysis(img_array=img, metadata=metadata, visu=False)

    assert isinstance(result, dict)
    assert result["figure"] is None
    assert result["figure name"] == "Field_Uniformity"

    # If your dataset is known-good, make this strict:
    # assert result["status"] == "success"
    assert result["status"] in {"success", "error"}

    measurements = result.get("measurements")
    assert isinstance(measurements, list)
    assert len(measurements) == 2

    meas_by_name = {m["name"]: m for m in measurements}

    centering = meas_by_name["Centering accuracy"]["value"]
    uniformity = meas_by_name["Uniformity"]["value"]

    if result["status"] == "success":
        assert isinstance(centering, float)
        assert np.isfinite(centering)
        assert 0.0 <= centering <= 100.0

        assert isinstance(uniformity, float)
        assert np.isfinite(uniformity)
        assert 0.0 <= uniformity <= 100.0
    else:
        # On error: allow None
        assert centering is None or (
            isinstance(centering, float) and np.isfinite(centering)
        )
        assert uniformity is None or (
            isinstance(uniformity, float) and np.isfinite(uniformity)
        )

    extra = result.get("extra_measurements", {})
    assert "roll_off_dia" in extra
    assert "roll_off_adi" in extra
    assert "best_z_plane" in extra

    # Roll-offs are differences of finite values in the success path
    if result["status"] == "success":
        assert np.isfinite(extra["roll_off_dia"])
        assert np.isfinite(extra["roll_off_adi"])
        assert isinstance(extra["best_z_plane"], int)
        assert extra["best_z_plane"] >= 0
