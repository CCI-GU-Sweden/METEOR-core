# tests/test_function.py

import METEOR_core.functions as f
import numpy as np


def test_norm_img_flat():
    img = np.ones((10, 10))
    out = f.norm_img(img)
    assert np.all(out == 0)


def test_dimension_slicer_basic():
    img = np.zeros((1, 2, 3, 4, 5))  # TCZYX
    metadata = {"axis_order": "TCZYX"}
    sliced = f.dimension_slicer(img, metadata, "CZYX")
    assert sliced.shape == (2, 3, 4, 5)


def test_sort_grid_identity():
    pts = np.array([[x, y] for x in range(3) for y in range(3)])
    sorted_pts, idx = f.sort_grid(pts, gridsize=3)
    assert sorted_pts.shape == (9, 2)
    assert len(idx) == 9


def test_aggregate_close_points_2d():
    pts = np.array([[0, 0], [0.4, 0.3], [5, 5]])
    agg, groups = f.aggregate_close_points(
        pts, diameter_um=1.0, twoD=True, return_groups=True
    )
    assert agg.shape[0] == 2
