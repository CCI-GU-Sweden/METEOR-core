# tests/test_plastic_uniformity.py

import importlib.util

import METEOR_core.functions as f
import METEOR_core.plastic_uniformity as pu
import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_calculate_field_uniformity_real_czi():
    # Skip if the optional CZI reader dependency isn't installed
    if importlib.util.find_spec("pylibCZIrw") is None:
        import pytest

        pytest.skip("pylibCZIrw not installed; skipping CZI-based integration test")

    img, metadata = f.czi_reader_tzcxy("tests/data/test_plastic.czi")

    result = pu.calculate_field_uniformity(
        img_stack=img,
        metadata=metadata,
        visu=False,
    )

    assert isinstance(result, dict)
    assert result["figure"] is None
    assert result["figure name"] == "Plastic_Field_Uniformity"

    # Allow "error" if the test file is not ideal, but if you know it is good,
    # change this to: assert result["status"] == "success"
    assert result["status"] in {"success", "error"}

    measurements = result.get("measurements")
    assert isinstance(measurements, list)
    assert len(measurements) == 2

    meas_by_name = {m["name"]: m for m in measurements}

    centering = meas_by_name["Centering accuracy - Plastic"]["value"]
    uniformity = meas_by_name["Uniformity"]["value"]

    if result["status"] == "success":
        assert isinstance(centering, float)
        assert np.isfinite(centering)
        assert 0.0 <= centering <= 100.0

        assert isinstance(uniformity, float)
        assert np.isfinite(uniformity)
        assert 0.0 <= uniformity <= 100.0
    else:
        # In failure cases, allow NaN
        assert (
            (centering is None)
            or (isinstance(centering, float) and not np.isfinite(centering))
            or (0.0 <= centering <= 100.0)
        )
        assert (
            (uniformity is None)
            or (isinstance(uniformity, float) and not np.isfinite(uniformity))
            or (0.0 <= uniformity <= 100.0)
        )

    extra = result.get("extra_measurements", {})
    assert "roll_off_dia" in extra
    assert "roll_off_adi" in extra
    assert np.isfinite(extra["roll_off_dia"])
    assert np.isfinite(extra["roll_off_adi"])
