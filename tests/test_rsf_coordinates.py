import METEOR_core.RSF as rsf
import numpy as np
import pytest
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

    def capture(volume, upsampling_factor, crop_size):
        captured.append(volume.copy())
        return None

    monkeypatch.setattr(rsf, "process_ring_crop", capture)
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
