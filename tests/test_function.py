# tests/test_function.py

import METEOR_core.functions as f
import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_czi_reader():
    # Real test with a small CZI file
    img, metadata = f.czi_reader_tzcxy("tests/data/test_intensity.czi")
    assert isinstance(img, np.ndarray)
    assert isinstance(metadata, dict)


def test_dimension_slicer_basic():
    img = np.zeros((1, 2, 3, 4, 5))  # TCZYX
    metadata = {"axis_order": "TCZYX"}
    sliced = f.dimension_slicer(img, metadata, "CZYX")
    assert sliced.shape == (2, 3, 4, 5)
