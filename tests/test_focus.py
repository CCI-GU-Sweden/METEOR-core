import METEOR_core.coregistration_analysis as coregistration
import METEOR_core.distortion_analysis as distortion
import METEOR_core.intensity_analysis as intensity
import METEOR_core.uniformity_analysis as uniformity
import numpy as np
import pytest
from METEOR_core.focus import select_focus_plane
from scipy.ndimage import gaussian_filter


def make_stack():
    image = np.zeros((64, 64), dtype=float)
    image[16:48, 16:48] = 1
    return np.stack([gaussian_filter(image, 3), image, gaussian_filter(image, 2)])


def test_focus_selects_sharp_plane_and_handles_ties():
    stack = make_stack()
    index, scores = select_focus_plane(stack)
    assert index == 1
    assert scores[1] > scores[0]
    assert select_focus_plane(np.stack([stack[1], stack[1]]))[0] == 0


@pytest.mark.parametrize("stack", [np.zeros((3, 8, 8)), np.full((2, 8, 8), np.nan)])
def test_focus_rejects_unusable_stacks(stack):
    with pytest.raises(ValueError):
        select_focus_plane(stack)


def test_ring_analyses_only_measure_selected_plane(monkeypatch):
    stack = make_stack()
    metadata = {"axis_order": "ZYX", "Image Size Z": 3}
    seen = []

    def fail_distortion(plane, *args, **kwargs):
        seen.append(plane)
        return None

    def fail_uniformity(plane, *args, **kwargs):
        seen.append(plane)
        return None

    monkeypatch.setattr(distortion, "get_distortion", fail_distortion)
    monkeypatch.setattr(uniformity, "get_uniformity", fail_uniformity)
    assert (
        distortion.distortion_analysis(stack, metadata, visu=False)["status"] == "error"
    )
    assert (
        uniformity.uniformity_analysis(stack, metadata, visu=False)["status"] == "error"
    )
    assert len(seen) == 2
    assert all(np.array_equal(plane, stack[1]) for plane in seen)


def test_coregistration_uses_first_channel_focus(monkeypatch):
    stack = make_stack()
    channels = np.stack([stack, stack[::-1]])
    metadata = {"axis_order": "CZYX", "Image Size Z": 3, "Image Size C": 2}
    seen = []

    def fail_analysis(planes, *args, **kwargs):
        seen.append(planes)
        return None

    monkeypatch.setattr(coregistration, "get_coregistration", fail_analysis)
    result = coregistration.coregistration_analysis(channels, metadata, visu=False)
    assert result["status"] == "error"
    assert len(seen) == 1
    assert np.array_equal(seen[0], channels[:, 1])


def test_intensity_uses_selected_focus_plane(monkeypatch):
    stack = make_stack()
    metadata = {"axis_order": "ZYX", "Image Size Z": 3, "Physical pixel size X": 1}
    seen = []

    def fake_detection(plane, *args):
        seen.append(plane)
        raise ValueError("test detection failure")

    monkeypatch.setattr(intensity, "_detect_pattern", fake_detection)
    result = intensity.intensity_analysis(stack, metadata)
    assert result["status"] == "error"
    assert len(seen) == 1
    assert np.array_equal(seen[0], stack[1])
