from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from config import AppConfig


def get_git_commit(base_dir: Optional[Path] = None) -> Optional[str]:
    try:
        # Recording the commit links a trained model to the source code that
        # produced it. The metadata still works outside a Git checkout.
        repo_root = base_dir or Path(__file__).resolve().parent
        result = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.strip() or None
    except Exception:
        return None


def build_training_metadata(
    config: AppConfig,
    model_history: Optional[Dict] = None,
) -> Dict:
    history = {}
    if model_history is not None:
        # Keras History objects expose metrics through `.history`; accepting a
        # plain dictionary also keeps this helper easy to test and reuse.
        raw_history = getattr(model_history, "history", model_history)
        for key, value in raw_history.items():
            history[key] = list(value)

    metadata = {
        "model_name": "emotion_model",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "framework": "tensorflow",
        "config": {
            "image_size": config.image_size,
            "emotion_labels": list(config.emotion_labels),
            "confidence_threshold": config.confidence_threshold,
            "face_padding": config.face_padding,
            "num_train": config.num_train,
            "num_val": config.num_val,
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "train_dir": str(config.train_dir),
            "val_dir": str(config.val_dir),
            "model_path": str(config.model_path),
        },
        "history": history,
        "git_commit": get_git_commit(config.base_dir),
    }
    return metadata


def save_training_metadata(
    config: AppConfig,
    model_history: Optional[Dict] = None,
) -> Path:
    artifacts_dir = config.base_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True, parents=True)

    metadata = build_training_metadata(config, model_history)
    manifest_path = artifacts_dir / "training_manifest.json"
    manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return manifest_path


def load_training_metadata(config: AppConfig) -> Dict:
    manifest_path = config.base_dir / "artifacts" / "training_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Training metadata not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))
