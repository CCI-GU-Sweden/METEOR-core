# tests/test_plastic_uniformity.py

import METEOR_core.plastic_uniformity as pu
import numpy as np


def test_calculate_field_uniformity_smoke(monkeypatch):
    # --- mock heavy helpers ---
    monkeypatch.setattr(pu, "uniformity", lambda img: 95.0)
    monkeypatch.setattr(pu, "centuring_acc", lambda img: 98.0)

    # Avoid expensive filtering / resizing
    monkeypatch.setattr(pu, "process_img", lambda imgs, _: np.ones((200, 200)))
    monkeypatch.setattr(pu, "bin_image", lambda img, _: img)

    img = np.ones((64, 64))
    metadata = {
        "Physical pixel size X": 0.1,
        "Image Size Z": 1,
        "Image Size C": 1,
        "Image Size T": 1,
    }

    result = pu.calculate_field_uniformity(
        img_stack=img,
        metadata=metadata,
        visu=False,
    )

    assert isinstance(result, dict)
    assert result["status"] == "success"
    assert result["figure"] is None

    measurements = result["measurements"]
    assert len(measurements) == 2

    centering = measurements[0]
    uniformity = measurements[1]

    assert centering["name"] == "Centering accuracy - Plastic"
    assert centering["value"] == 98.0

    assert uniformity["name"] == "Uniformity"
    assert uniformity["value"] == 95.0

    extra = result["extra_measurements"]
    assert "roll_off_dia" in extra
    assert "roll_off_adi" in extra
