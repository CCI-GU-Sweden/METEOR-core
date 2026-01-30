import importlib.util

import METEOR_core.coregistration_analysis as ca
import METEOR_core.functions as f
import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_coregistration_analysis_real_czi():
    # Skip if the optional CZI reader dependency isn't installed
    if importlib.util.find_spec("pylibCZIrw") is None:
        import pytest

        pytest.skip("pylibCZIrw not installed; skipping CZI-based integration test")

    img, metadata = f.czi_reader_tzcxy("tests/data/test_2drings.czi")

    result = ca.coregistration_analysis(img, metadata, visu=False)

    assert isinstance(result, dict)
    assert result["status"] == "success"

    measurements = result.get("measurements")
    assert isinstance(measurements, list)
    assert len(measurements) == 2

    meas_by_name = {m["name"]: m for m in measurements}

    pearson = meas_by_name["Pearson"]["value"]
    amplitude = meas_by_name["Amplitude"]["value"]

    assert isinstance(pearson, float)
    assert np.isfinite(pearson)
    assert -1.0 <= pearson <= 1.0

    assert isinstance(amplitude, float)
    assert np.isfinite(amplitude)
    assert amplitude >= 0.0
