# tests/test_distortion_analysis.py

import importlib.util

import METEOR_core.distortion_analysis as da
import METEOR_core.functions as f
import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_distortion_analysis_real_czi():
    # Skip if the optional CZI reader dependency isn't installed
    if importlib.util.find_spec("pylibCZIrw") is None:
        import pytest

        pytest.skip("pylibCZIrw not installed; skipping CZI-based integration test")

    img, metadata = f.czi_reader_tzcxy("tests/data/test_2drings.czi")

    result = da.distortion_analysis(img_array=img, metadata=metadata, visu=False)

    assert isinstance(result, dict)
    assert result["status"] == "success"
    assert result["figure"] is None

    measurements = result.get("measurements")
    assert isinstance(measurements, list)
    assert len(measurements) == 1

    meas = measurements[0]
    assert meas["category"] == "Distortion"
    assert meas["name"] == "Amplitude"

    amp_um = meas["value"]
    assert isinstance(amp_um, float)
    assert np.isfinite(amp_um)
    assert amp_um >= 0.0

    extra = result.get("extra_measurements", {})
    assert "max_vector_amplitude_pxl" in extra
    assert np.isfinite(extra["max_vector_amplitude_pxl"])
    assert extra["max_vector_amplitude_pxl"] >= 0.0

    assert "geometric_distortion_rate" in extra
    assert np.isfinite(extra["geometric_distortion_rate"])

    assert "best_z_plane" in extra
    assert isinstance(extra["best_z_plane"], int)
    assert extra["best_z_plane"] >= 0
