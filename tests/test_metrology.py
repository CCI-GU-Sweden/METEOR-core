# tests/test_metrology.py

import METEOR_core.metrology as metrology
import numpy as np


def test_by_name_basic():
    assert metrology.by_name("my psf image.tif") == "Psf_Measure"
    assert metrology.by_name("Argo_2D_Rings_01") == "Argo_2D_Rings_Analysis"
    assert metrology.by_name("unknown stuff") == "Unknown"


def test_classify_falls_back_to_unknown():
    metadata = {}
    assert metrology.classify("completely random", metadata) == "Unknown"


def make_dummy_result():
    # Minimal fake result that mimics your documented structure
    return {
        "figure": None,
        "figure name": "dummy.png",
        "status": "success",
        "measurements": [
            {
                "category": "PSF",
                "name": "FWHM XY",
                "value": 0.2,
                "unit": "microns",
            }
        ],
        "extra_measurements": {"snr": 42.0},
    }


def test_metrology_splitter_argo_2d_rings(monkeypatch):
    # Arrange: mock the various analysis functions so we don't do real work
    dummy_result = make_dummy_result()

    def fake_fn(img, metadata):
        # You could add assertions on img/metadata here if you want
        return dummy_result

    monkeypatch.setattr(metrology, "coregistration_analysis", fake_fn)
    monkeypatch.setattr(metrology, "distortion_analysis", fake_fn)
    monkeypatch.setattr(metrology, "uniformity_analysis", fake_fn)
    monkeypatch.setattr(metrology, "RSF_analysis", fake_fn)

    # A minimal "image" and metadata
    img = np.zeros((10, 10))
    metadata = {"Image Size T": 1, "Image Size Z": 1}

    # Act
    results = metrology.metrology_splitter(
        img, "Argo_2D_Rings_01", metadata, category_override="2d rings"
    )

    # Assert
    assert isinstance(results, list)
    assert len(results) == 4
    for res in results:
        assert isinstance(res, dict)
        assert "figure" in res
        assert "figure name" in res
        assert "status" in res
        assert "measurements" in res
        assert "extra_measurements" in res


def test_metrology_splitter_unknown_category_returns_false(monkeypatch):
    # Because of the NameError branch, it should return False for unknown.
    # We mock logger.warning so the test doesn't depend on logging config.
    monkeypatch.setattr(metrology.logger, "warning", lambda *args, **kwargs: None)

    img = np.zeros((5, 5))
    metadata = {}
    result = metrology.metrology_splitter(img, "totally_unknown_name", metadata)
    assert result is False
