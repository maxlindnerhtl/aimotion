import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import AppConfig
from reproducibility import build_training_metadata, save_training_metadata


def test_build_training_metadata_includes_expected_fields(tmp_path):
    config = AppConfig(base_dir=tmp_path)
    metadata = build_training_metadata(config, {"accuracy": [0.91], "val_accuracy": [0.88]})

    assert metadata["model_name"] == "emotion_model"
    assert metadata["config"]["image_size"] == config.image_size
    assert metadata["history"]["accuracy"] == [0.91]


def test_save_training_metadata_writes_manifest(tmp_path):
    config = AppConfig(base_dir=tmp_path)
    manifest_path = save_training_metadata(config, {"accuracy": [0.9]})

    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["config"]["num_epochs"] == config.num_epochs
    assert payload["history"]["accuracy"] == [0.9]
