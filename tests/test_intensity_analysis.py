# tests/test_intensity_analysis.py

import METEOR_core.intensity_analysis as ia
import numpy as np


def test_intensity_analysis_smoke(monkeypatch):
    # --- mock get_intensity to avoid heavy processing ---
    def fake_get_intensity(*args, **kwargs):
        return 0.98, None  # (r2, fig)

    monkeypatch.setattr(ia, "get_intensity", fake_get_intensity)

    img = np.zeros((64, 64))
    metadata = {
        "Image Size Z": 1,
        "Image Size C": 1,
        "Image Size T": 1,
        "Physical pixel size X": 0.1,
    }

    result = ia.intensity_analysis(img, metadata)

    assert isinstance(result, dict)
    assert result["status"] == "success"

    meas = result["measurements"][0]
    assert meas["category"] == "Intensity"
    assert meas["name"] == "Linearity of Intensity"
    assert meas["value"] == 0.98

    assert "figure" in result
    assert "extra_measurements" in result


def test_intensity_analysis_multiz_selects_best(monkeypatch):
    values = [0.2, None, 0.9]

    def fake_get_intensity(img, *_, **__):
        return values.pop(0), None

    monkeypatch.setattr(ia, "get_intensity", fake_get_intensity)

    img = np.zeros((3, 32, 32))  # ZYX
    metadata = {
        "Image Size Z": 3,
        "Image Size C": 1,
        "Image Size T": 1,
        "Physical pixel size X": 0.1,
    }

    result = ia.intensity_analysis(img, metadata)
    assert result["status"] == "success"
    assert result["extra_measurements"]["best_z_plane"] == 2
