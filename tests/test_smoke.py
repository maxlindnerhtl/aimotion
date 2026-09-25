import sys
from pathlib import Path

import cv2
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import AppConfig, resolve_path
from model_utils import build_model, load_overlay


def test_resolve_path_uses_base_dir_for_relative_values(tmp_path):
    config = AppConfig(base_dir=tmp_path)
    assert resolve_path("model.h5", tmp_path) == tmp_path / "model.h5"
    assert config.model_path == tmp_path / "model.h5"


def test_build_model_has_expected_output_shape():
    config = AppConfig()
    model = build_model(config)
    sample = np.random.rand(1, config.image_size, config.image_size, 1).astype("float32")
    prediction = model(sample)
    assert prediction.shape == (1, 7)


def test_load_overlay_accepts_bgr_image(tmp_path):
    config = AppConfig(base_dir=tmp_path)
    overlay_path = tmp_path / "overlay.png"
    image = np.zeros((10, 20, 3), dtype=np.uint8)

    cv2.imwrite(str(overlay_path), image)

    overlay = load_overlay(str(overlay_path), config)
    assert overlay.shape[2] == 4
