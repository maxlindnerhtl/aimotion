from __future__ import annotations

import cv2
import numpy as np
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import AppConfig, resolve_path


def build_model(config: AppConfig) -> Sequential:
    # The final layer must have one output per configured expression label.
    # Its softmax values are the class scores used during inference.
    model = Sequential(
        [
            Input(shape=(config.image_size, config.image_size, 1)),
            Conv2D(32, (3, 3), activation="relu"),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(1024, activation="relu"),
            Dropout(0.5),
            Dense(len(config.emotion_labels), activation="softmax"),
        ]
    )
    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.0001), metrics=["accuracy"])
    return model


def load_overlay(path_value, config: AppConfig):
    overlay_path = resolve_path(path_value, config.base_dir)
    if overlay_path is None:
        overlay_path = config.default_overlay_path

    if not overlay_path.exists():
        raise FileNotFoundError(f"Overlay image not found: {overlay_path}")

    overlay = cv2.imread(str(overlay_path), cv2.IMREAD_UNCHANGED)
    if overlay is None:
        raise ValueError(f"Could not load overlay image: {overlay_path}")

    # Normalize three-channel images to BGRA so the display loop can always
    # apply an alpha channel, regardless of the input PNG format.
    if overlay.shape[2] == 3:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

    return overlay


def create_data_generators(config: AppConfig):
    # Training and validation use identical normalization, but validation does
    # not receive augmentation so its metrics remain comparable between runs.
    train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        str(config.train_dir),
        target_size=(config.image_size, config.image_size),
        batch_size=config.batch_size,
        color_mode="grayscale",
        class_mode="categorical",
    )

    validation_generator = val_datagen.flow_from_directory(
        str(config.val_dir),
        target_size=(config.image_size, config.image_size),
        batch_size=config.batch_size,
        color_mode="grayscale",
        class_mode="categorical",
    )

    return train_generator, validation_generator
