import numpy as np
import pytest
from METEOR_core.focus import _stack_percentiles
from METEOR_core.functions import norm_img
from METEOR_core.ring_lattice import match_points
from METEOR_core.RSF import _average_ring_crops, process_ring_crop
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage import filters, measure


@pytest.mark.parametrize(
    "shape,factor", [((3, 15, 15), 3), ((4, 20, 20), 5), ((5, 17, 19), 2)]
)
def test_rsf_crop_matches_original_zoom(shape, factor):
    z, y, x = np.indices(shape)
    image = (
        np.exp(-((y - shape[1] * 0.48) ** 2 + (x - shape[2] * 0.53) ** 2) / 15)
        * (1 + z / 10)
        * 500
    ).astype(np.uint16)
    crop_size = min(shape[1:]) // 2
    zoomed = norm_img(ndimage.zoom(image, (factor,) * 3, order=0))
    mid = zoomed[zoomed.shape[0] // 2]
    labels = measure.label(mid > filters.threshold_otsu(mid))
    region = max(measure.regionprops(labels, intensity_image=mid), key=lambda r: r.area)
    cy, cx = region.weighted_centroid
    half = int(crop_size * factor * 0.2)
    expected = filters.median(
        zoomed[
            :,
            max(0, int(cy) - half) : int(cy) + half,
            max(0, int(cx) - half) : int(cx) + half,
        ]
    )
    actual = process_ring_crop(image, factor, crop_size)
    np.testing.assert_allclose(actual, expected, atol=0, rtol=0)


def test_histogram_percentiles_equal_numpy_for_integer_stacks():
    rng = np.random.default_rng(4)
    for dtype in (np.uint8, np.uint16, np.int16):
        stack = rng.integers(
            -100 if dtype == np.int16 else 0, 200, size=(3, 51, 49)
        ).astype(dtype)
        np.testing.assert_allclose(
            _stack_percentiles(stack), np.percentile(stack, [1, 99.9])
        )


def test_rsf_shape_accumulator_fallback_preserves_modal_average():
    centers = np.column_stack((np.arange(5), np.zeros(5)))

    def process(y, x):
        size = 4 if y in (0, 2, 4) else 5
        return (y, x), np.full((3, size, size), y, dtype=float)

    reference = _average_ring_crops(centers, process, max_accumulator_bytes=10_000)
    bounded = _average_ring_crops(centers, process, max_accumulator_bytes=1)
    np.testing.assert_array_equal(reference[0], bounded[0])
    np.testing.assert_array_equal(reference[1], bounded[1])
    assert np.all(bounded[1] == 2)


def test_sparse_assignment_matches_dense_objective():
    rng = np.random.default_rng(5)
    for _ in range(10):
        refs, detections = rng.random((25, 2)), rng.random((20, 2))
        gate = 0.25
        distances = cdist(refs, detections)
        costs = np.concatenate(
            (distances, np.full((len(refs), len(refs)), gate)), axis=1
        )
        rows, cols = linear_sum_assignment(costs)
        dense_cost = costs[rows, cols].sum()
        sparse = match_points(refs, detections, gate)
        sparse_cost = (
            distances[sparse[:, 0], sparse[:, 1]].sum()
            + (len(refs) - len(sparse)) * gate
        )
        assert sparse_cost == pytest.approx(dense_cost)
