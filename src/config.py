from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class AppConfig:
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    config_path: Optional[Path] = None
    model_path: Path = field(init=False)
    cascade_path: Path = field(init=False)
    default_overlay_path: Path = field(init=False)
    data_dir: Path = field(init=False)
    train_dir: Path = field(init=False)
    val_dir: Path = field(init=False)
    recording_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "recordings")
    snapshot_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "snapshots")

    image_size: int = 48
    camera_width: int = 1280
    camera_height: int = 720
    overlay_enabled: bool = True
    show_confidence: bool = True
    emotion_labels: Tuple[str, ...] = (
        "Angry",
        "Disgusted",
        "Fear",
        "Happy",
        "Neutral",
        "Sad",
        "Surprised",
    )
    confidence_threshold: float = 0.45
    face_padding: float = 0.15
    num_train: int = 28709
    num_val: int = 7178
    batch_size: int = 64
    num_epochs: int = 50

    @classmethod
    def from_file(
        cls,
        path: Optional[str] = None,
        base_dir: Optional[Path] = None,
    ) -> "AppConfig":
        base = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
        config_file = Path(path) if path else base / "config.json"
        overrides = {}

        # The JSON file is intentionally an override layer: keeping defaults in
        # Python means the application can still start with a partial config.
        if config_file.exists():
            with open(config_file, encoding="utf-8") as cfg:
                data = json.load(cfg)
            if isinstance(data, dict):
                for key in (
                    "image_size",
                    "camera_width",
                    "camera_height",
                    "overlay_enabled",
                    "show_confidence",
                    "confidence_threshold",
                    "face_padding",
                    "num_train",
                    "num_val",
                    "batch_size",
                    "num_epochs",
                ):
                    if key in data:
                        overrides[key] = data[key]

                if "recording_dir" in data:
                    overrides["recording_dir"] = Path(data["recording_dir"])
                if "snapshot_dir" in data:
                    overrides["snapshot_dir"] = Path(data["snapshot_dir"])

        if "recording_dir" not in overrides:
            overrides["recording_dir"] = base / "recordings"
        if "snapshot_dir" not in overrides:
            overrides["snapshot_dir"] = base / "snapshots"

        # Resolve output directories relative to the selected config location,
        # not relative to whichever directory the user launched the command from.
        for dir_key in ("recording_dir", "snapshot_dir"):
            value = overrides[dir_key]
            if not isinstance(value, Path):
                value = Path(value)
            if not value.is_absolute():
                value = base / value
            overrides[dir_key] = value

        return cls(base_dir=base, config_path=config_file, **overrides)

    def __post_init__(self):
        # Asset paths are derived from one base directory so training and
        # inference use the same model, cascade, and data locations.
        object.__setattr__(self, "model_path", self.base_dir / "model.h5")
        object.__setattr__(self, "cascade_path", self.base_dir / "haarcascade_frontalface_default.xml")
        object.__setattr__(self, "default_overlay_path", self.base_dir / "overlay.png")
        object.__setattr__(self, "data_dir", self.base_dir / "data")
        object.__setattr__(self, "train_dir", self.data_dir / "train")
        object.__setattr__(self, "val_dir", self.data_dir / "test")

        if not isinstance(self.recording_dir, Path):
            object.__setattr__(self, "recording_dir", Path(self.recording_dir))
        if not isinstance(self.snapshot_dir, Path):
            object.__setattr__(self, "snapshot_dir", Path(self.snapshot_dir))

        default_root = Path(__file__).resolve().parent
        if self.recording_dir == default_root / "recordings" and self.base_dir != default_root:
            object.__setattr__(self, "recording_dir", self.base_dir / "recordings")
        elif not self.recording_dir.is_absolute():
            object.__setattr__(self, "recording_dir", self.base_dir / self.recording_dir)

        if self.snapshot_dir == default_root / "snapshots" and self.base_dir != default_root:
            object.__setattr__(self, "snapshot_dir", self.base_dir / "snapshots")
        elif not self.snapshot_dir.is_absolute():
            object.__setattr__(self, "snapshot_dir", self.base_dir / self.snapshot_dir)


def resolve_path(path_value, base_dir: Optional[Path] = None) -> Optional[Path]:
    if path_value is None:
        return None

    # Command-line paths may be relative to the source/config directory.
    base = base_dir or Path(__file__).resolve().parent
    path = Path(path_value)
    if not path.is_absolute():
        path = base / path
    return path
