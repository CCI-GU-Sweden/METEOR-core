from types import SimpleNamespace

import METEOR_core.RSF as rsf
import numpy as np
import pytest
from skimage import measure
from skimage.draw import disk
from skimage.transform import rotate


@pytest.mark.parametrize("angle, offset", [(0, (0, 0)), (17, (12, -9))])
def test_rsf_extracts_detected_features_in_original_coordinates(
    monkeypatch, angle, offset
):
    image = np.zeros((160, 180), dtype=float)
    # Different areas keep the middle-sized feature after the area filter.
    for center, radius in [((45, 50), 3), ((80, 90), 5), ((115, 130), 7)]:
        rr, cc = disk(center, radius, shape=image.shape)
        image[rr, cc] = 1
    image = rotate(image, angle, order=0, preserve_range=True)
    image = np.roll(image, offset, axis=(0, 1))
    stack = np.stack([image * 0.5, image, image * 0.5])
    captured = []

    def capture(volume, upsampling_factor, crop_size, **kwargs):
        captured.append(volume.copy())
        return None

    monkeypatch.setattr(rsf, "process_ring_crop", capture)
    regions = measure.regionprops(measure.label(image > 0))
    points = np.asarray([(r.centroid[1], r.centroid[0]) for r in regions])
    registration = SimpleNamespace(
        points_xy=points,
        indices=np.array([[0, 0], [1, 1], [2, 2]]),
        spacing_px=40,
        diagnostics={"detected_count": len(points), "index_extent": [3, 3]},
    )
    monkeypatch.setattr(rsf, "select_focus_plane", lambda stack: (1, [1, 2, 1]))
    monkeypatch.setattr(
        rsf, "detect_ring_lattice", lambda *args, **kwargs: registration
    )
    metadata = {
        "axis_order": "ZYX",
        "Image Size Z": 3,
        "Physical pixel size X": 0.5,
        "Physical pixel size Z": 0.5,
    }
    rsf.RSF_analysis(stack, metadata, visu=False, extracrop_f=4)

    assert captured
    for volume in captured:
        cy, cx = np.array(volume.shape[1:]) // 2
        assert volume[1, cy, cx] == 1


def test_rsf_reports_excess_candidates_and_excludes_grid_center(monkeypatch):
    indices = np.array([[i, j] for i in range(3) for j in range(3)])
    points = indices * 10 + 20
    registration = SimpleNamespace(
        points_xy=points,
        indices=indices,
        spacing_px=10,
        diagnostics={"detected_count": 10, "index_extent": [3, 3]},
    )
    seen = []
    monkeypatch.setattr(rsf, "select_focus_plane", lambda stack: (0, [1]))
    monkeypatch.setattr(
        rsf, "detect_ring_lattice", lambda *args, **kwargs: registration
    )
    monkeypatch.setattr(
        rsf,
        "process_ring_crop",
        lambda volume, *args, **kw: seen.append(volume) or None,
    )
    result = rsf.RSF_analysis(
        np.ones((1, 80, 80)),
        {"axis_order": "ZYX", "Physical pixel size X": 0.5},
        grid_size=3,
        visu=False,
        extracrop_f=2,
    )
    assert len(seen) == 8
    assert "10 candidate" in result["extra_measurements"]["warnings"][0]


def test_local_ring_search_does_not_pick_a_brighter_neighbor():
    image = np.zeros((3, 40, 40))
    rr, cc = disk((20, 20), 2, shape=image.shape[1:])
    image[:, rr, cc] = 10
    rr, cc = disk((20, 32), 4, shape=image.shape[1:])
    image[:, rr, cc] = 100
    result = rsf.process_ring_crop(image, 2, 20, reference_z=1, max_center_offset_px=5)
    assert result is not None
    assert 0 < np.max(result) < 0.5
