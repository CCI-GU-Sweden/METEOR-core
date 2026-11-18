import numpy as np
import pytest
from METEOR_core import metrology
from METEOR_core.classification import INTENSITY, PSF, RINGS, STAR, classify_image


def test_unknown_image_does_not_route_by_filename():
    result = classify_image(np.zeros((32, 32)), {}, name="2d_rings.czi")
    assert result["category"] == "Unknown"
    assert result["requires_review"]
    assert any(e.get("category") == RINGS for e in result["evidence"])


def test_explicit_override_and_structured_metadata():
    image = np.zeros((16, 16))
    result = classify_image(image, {}, override="beads")
    assert result["category"] == PSF
    assert not result["requires_review"]
    result = classify_image(image, {"pattern_type": "crossing stairs"})
    assert result["category"] == "3D_Crossing_Stairs"
    assert result["source"] == "metadata"
    with pytest.raises(ValueError):
        classify_image(image, {}, override="unsupported")


def test_metadata_conflict_and_multiscene_require_review():
    image = np.zeros((2, 16, 16))
    conflict = classify_image(
        image[0], {"pattern_type": "star", "analysis_type": "beads"}
    )
    assert conflict["category"] == "Unknown"
    scenes = classify_image(image, {"axis_order": "SYX", "pattern_type": "star"})
    assert scenes["requires_review"]


def test_conflicting_filename_does_not_override_image(monkeypatch):
    import METEOR_core.classification as classification

    monkeypatch.setattr(
        classification, "_star_evidence", lambda image: {"radial_peaks": 12}
    )
    image = np.arange(256, dtype=float).reshape(16, 16)
    result = classify_image(image, {}, "psf.czi")
    assert result["category"] == STAR
    assert any("Filename hint conflicts" in x for x in result["warnings"])

    conflict = classify_image(image, {"pattern_type": "psf"}, "star.czi")
    assert conflict["requires_review"]
    assert conflict["category"] == "Unknown"
    assert any("conflicts with image" in x for x in conflict["warnings"])


def test_sparse_beads_are_only_suggestive(monkeypatch):
    import METEOR_core.classification as classification

    monkeypatch.setattr(classification, "_star_evidence", lambda image: None)
    monkeypatch.setattr(
        classification,
        "_sparse_bead_evidence",
        lambda image, z: {"small_components": 40},
    )
    image = np.stack([np.arange(256).reshape(16, 16)] * 3)
    result = classify_image(image, {"axis_order": "ZYX"}, "beads.czi")
    assert result["requires_review"]
    assert result["candidates"][0]["category"] == PSF
    assert result["candidates"][0]["strength"] == "suggestive"


def test_ring_dispatch_preserves_other_results_when_one_analysis_fails(monkeypatch):
    def failure(img, metadata):
        raise ValueError("requires two channels")

    def success(img, metadata):
        return {"status": "success"}

    monkeypatch.setattr(metrology, "coregistration_analysis", failure)
    monkeypatch.setattr(metrology, "distortion_analysis", success)
    monkeypatch.setattr(metrology, "uniformity_analysis", success)
    monkeypatch.setattr(metrology, "RSF_analysis", success)
    results = metrology.metrology_splitter(
        np.ones((8, 8)), "", {}, category_override=RINGS
    )
    assert [r["status"] for r in results] == ["error", "success", "success", "success"]


def test_splitter_returns_classification_with_override(monkeypatch):
    monkeypatch.setattr(
        metrology, "psf_analysis", lambda img, metadata: {"status": "success"}
    )
    results = metrology.metrology_splitter(
        np.ones((8, 8)), "unknown", {}, category_override="psf"
    )
    assert results[0]["classification"]["source"] == "override"


@pytest.mark.integration
@pytest.mark.parametrize(
    "path, category",
    [
        ("tests/data/test_star.czi", STAR),
        ("tests/data/test_intensity.czi", INTENSITY),
        ("tests/data/test_2drings.czi", RINGS),
    ],
)
def test_classify_existing_czi_without_filename(path, category):
    pytest.importorskip("pylibCZIrw")
    from METEOR_core.functions import czi_reader_tzcxy

    img, metadata = czi_reader_tzcxy(path)
    result = classify_image(img, metadata, "anonymous.czi")
    assert result["category"] == category, result
    assert not result["requires_review"]
