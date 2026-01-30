# tests/test_coregistration_analysis.py

import METEOR_core.coregistration_analysis as ca
import numpy as np


def test_coregistration_analysis_smoke(monkeypatch):
    # --- fake get_coregistration ---
    fake_result = {
        "max_amplitude": 2.0,
        "red_c": np.ones((10, 10)),
        "blue_c": np.ones((10, 10)),
        "blue_points": np.array([[0, 0], [1, 1]]),
        "red_points": np.array([[0, 0], [0.9, 1.1]]),
    }

    monkeypatch.setattr(ca, "get_coregistration", lambda *args, **kwargs: fake_result)
    monkeypatch.setattr(ca.measure, "pearson_corr_coeff", lambda a, b: (0.95, None))
    monkeypatch.setattr(ca, "sort_grid", lambda pts: (None, np.arange(len(pts))))

    img = np.zeros((2, 3, 10, 10))  # CZYX
    metadata = {
        "Physical pixel size X": 0.1,
        "Image Size C": 2,
        "Image Size Z": 3,
        "Image Size T": 1,
        "axis_order": "CZYX",
    }

    result = ca.coregistration_analysis(
        img_array=img,
        metadata=metadata,
        visu=False,
    )

    assert result["status"] == "success"
    assert result["figure"] is None

    measurements = result["measurements"]
    assert len(measurements) == 2

    pearson = measurements[0]
    amplitude = measurements[1]

    assert pearson["name"] == "Pearson"
    assert pearson["value"] == 0.95

    assert amplitude["name"] == "Amplitude"
    assert amplitude["value"] == 2.0 * 0.1  # converted to µm

    extra = result["extra_measurements"]
    assert extra["pearson_coeff"] == 0.95
    assert extra["max_vector_amplitude_pxl"] == 2.0
