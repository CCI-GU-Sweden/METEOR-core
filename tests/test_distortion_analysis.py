# tests/test_distortion_analysis.py

import METEOR_core.distortion_analysis as da
import numpy as np


def test_distortion_analysis_smoke(monkeypatch):
    fake_distortion = {
        "paired_lattice": np.array([[0.0, 0.0], [1.0, 1.0]]),
        "points": np.array([[0.1, 0.0], [1.1, 1.0]]),
        "max_amplitude": 1.0,
        "img": np.ones((10, 10)),
    }

    # --- mock heavy functions ---
    monkeypatch.setattr(da, "get_distortion", lambda *args, **kwargs: fake_distortion)
    monkeypatch.setattr(da, "dimension_slicer", lambda img, *_: img)

    img = np.zeros((3, 10, 10))  # ZYX
    metadata = {
        "Physical pixel size X": 0.2,
        "Image Size Z": 3,
        "Image Size C": 1,
        "Image Size T": 1,
        "axis_order": "ZYX",
    }

    result = da.distortion_analysis(
        img_array=img,
        metadata=metadata,
        visu=False,
    )

    assert result["status"] == "success"
    assert result["figure"] is None

    meas = result["measurements"][0]
    assert meas["category"] == "Distortion"
    assert meas["name"] == "Amplitude"
    assert round(meas["value"], 5) == 1.0 * 0.2  # µm conversion

    extra = result["extra_measurements"]
    assert extra["max_vector_amplitude_pxl"] == 1.0
    assert "geometric_distortion_rate" in extra
    assert extra["best_z_plane"] == 0
