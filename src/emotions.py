import argparse
import os
import time
from collections import Counter, deque
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from config import AppConfig, resolve_path
from model_utils import build_model, create_data_generators, load_overlay
from reproducibility import save_training_metadata


def draw_text(frame, text, position, scale=0.7, color=(255, 255, 255), thickness=2):
    """Draw anti-aliased TrueType text with a subtle shadow."""
    font_size = max(12, int(scale * 32))
    font_paths = (
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    )
    font = None
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    # Pillow renders TrueType fonts with smoother curves than OpenCV's
    # built-in bitmap fonts. Convert only for the duration of text drawing.
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    x, y = position
    rgb_color = (color[2], color[1], color[0])
    shadow_width = max(1, thickness)
    draw.text(
        (x + shadow_width, y - font_size + shadow_width),
        text,
        font=font,
        fill=(0, 0, 0),
        stroke_width=shadow_width,
        stroke_fill=(0, 0, 0),
    )
    draw.text(
        (x, y - font_size),
        text,
        font=font,
        fill=rgb_color,
        stroke_width=0,
    )
    frame[:, :] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def set_camera_resolution(cap, width, height):
    """Request a camera mode and return whether the device accepted it."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return abs(actual_width - width) <= 16 and abs(actual_height - height) <= 16


def plot_model_history(model_history: Any, config: AppConfig) -> None:
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    train_acc = model_history.history.get("accuracy", [])
    val_acc = model_history.history.get("val_accuracy", [])
    train_loss = model_history.history.get("loss", [])
    val_loss = model_history.history.get("val_loss", [])

    if train_acc:
        axs[0].plot(range(1, len(train_acc) + 1), train_acc)
    if val_acc:
        axs[0].plot(range(1, len(val_acc) + 1), val_acc)
    axs[0].set_title("Model Accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("Epoch")
    axs[0].legend(["train", "val"], loc="best")

    if train_loss:
        axs[1].plot(range(1, len(train_loss) + 1), train_loss)
    if val_loss:
        axs[1].plot(range(1, len(val_loss) + 1), val_loss)
    axs[1].set_title("Model Loss")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")
    fig.tight_layout()
    # Keep the plot next to the model so training outputs remain together.
    fig.savefig(config.base_dir / "plot.png")
    plt.show()


def train_model(config: Optional[AppConfig] = None) -> None:
    config = config or AppConfig()
    if not config.train_dir.exists() or not config.val_dir.exists():
        raise FileNotFoundError(f"Training data directories not found under: {config.data_dir}")

    train_generator, validation_generator = create_data_generators(config)
    model = build_model(config)
    # The generator reads images from the class folders and supplies batches
    # in the format expected by the grayscale CNN.
    model_info = model.fit(
        train_generator,
        steps_per_epoch=config.num_train // config.batch_size,
        epochs=config.num_epochs,
        validation_data=validation_generator,
        validation_steps=config.num_val // config.batch_size,
    )
    manifest_path = save_training_metadata(config, model_info)
    plot_model_history(model_info, config)
    model.save_weights(str(config.model_path))
    print(f"Model saved to: {config.model_path}")
    print(f"Training metadata saved to: {manifest_path}")


def run_display(
    overlay_value: Optional[str],
    camera_index: int = 0,
    config: Optional[AppConfig] = None,
    input_source: Optional[str] = None,
) -> None:
    config = config or AppConfig()
    if not config.model_path.exists():
        raise FileNotFoundError(f"Model file not found: {config.model_path}")

    model = build_model(config)
    model.load_weights(str(config.model_path))

    cv2.ocl.setUseOpenCL(False)

    cascade = cv2.CascadeClassifier(str(config.cascade_path))
    if cascade.empty():
        raise FileNotFoundError(f"Face cascade file not found or invalid: {config.cascade_path}")

    # A file source enables camera-free demos and automated playback. Webcam
    # frames are mirrored later because that matches the user's view.
    if input_source:
        source_path = resolve_path(input_source, config.base_dir)
        if source_path is None or not source_path.exists():
            raise FileNotFoundError(f"Input source not found: {input_source}")
        cap = cv2.VideoCapture(str(source_path))
        source_type = "file"
    else:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Warning: camera index {camera_index} could not be opened. Retrying with camera index 0...")
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam. Please ensure a camera is connected and available.")
        source_type = "camera"

    current_resolution = (config.camera_width, config.camera_height)
    if source_type == "camera":
        if not set_camera_resolution(cap, *current_resolution):
            current_resolution = (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            print(
                f"Camera selected {current_resolution[0]}x{current_resolution[1]} "
                "instead of the requested resolution."
            )
        # Keep the capture queue short so processing delays do not accumulate
        # into visible latency. Some camera backends ignore this setting.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # These values can be changed live with keyboard shortcuts without
    # modifying the persisted configuration file.
    overlay_enabled = config.overlay_enabled
    show_confidence = config.show_confidence
    recording = False
    video_writer = None
    config.recording_dir.mkdir(parents=True, exist_ok=True)
    config.snapshot_dir.mkdir(parents=True, exist_ok=True)

    overlay = None
    overlay_width = 0
    overlay_height = 0
    if overlay_value is not None or config.default_overlay_path.exists():
        overlay = load_overlay(overlay_value, config)
        overlay_width = 400
        overlay_height = int((overlay.shape[0] / overlay.shape[1]) * overlay_width)
        overlay = cv2.resize(overlay, (overlay_width, overlay_height), interpolation=cv2.INTER_AREA)

    window_name = "Video"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Apply fullscreen and topmost properties immediately so the camera view
    # opens as the primary window instead of appearing behind other windows.
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    if hasattr(cv2, "WND_PROP_TOPMOST"):
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    displayed_emotion = None
    label_history = deque(maxlen=5)
    last_seen_face = time.time()
    fps = 0.0
    show_help = True

    while True:
        start_time = time.time()
        try:
            ret, frame = cap.read()
        except cv2.error as error:
            if source_type != "camera":
                raise
            print(f"Camera capture failed: {error}")
            cap.release()
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                raise RuntimeError("Could not reopen the webcam after the capture failure.") from error
            set_camera_resolution(cap, *current_resolution)
            continue
        if not ret:
            if source_type == "file":
                break
            print("Failed to read frame from webcam. Retrying...")
            time.sleep(0.5)
            if not cap.isOpened():
                break
            continue

        if source_type == "camera":
            frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        current_face_label = None

        for (x, y, w, h) in faces:
            # Add a small margin because the expression model performs better
            # when it receives some context around the detected face.
            pad_x = int(w * config.face_padding)
            pad_y = int(h * config.face_padding)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(frame.shape[1], x + w + pad_x)
            y2 = min(frame.shape[0], y + h + pad_y)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (192, 192, 192), 2)

            roi_gray = gray[y1:y2, x1:x2]
            roi_resized = cv2.resize(roi_gray, (config.image_size, config.image_size), interpolation=cv2.INTER_AREA)
            roi_normalized = roi_resized.astype(np.float32) / 255.0
            roi_batch = roi_normalized[:, :, np.newaxis][np.newaxis, ...]

            # Inference expects a batch dimension and a single grayscale
            # channel: (1, image_size, image_size, 1).
            prediction = model(roi_batch, training=False).numpy()[0]
            max_index = int(np.argmax(prediction))
            confidence = float(prediction[max_index])
            label = config.emotion_labels[max_index]

            if confidence >= config.confidence_threshold:
                current_face_label = label
                label_history.append(label)
                # Majority voting over recent frames reduces flicker caused by
                # small changes in lighting or face position.
                if len(label_history) >= 2:
                    smoothed_label, _ = Counter(label_history).most_common(1)[0]
                    if displayed_emotion is None:
                        displayed_emotion = smoothed_label
                    elif smoothed_label != displayed_emotion and label_history.count(smoothed_label) >= 2:
                        displayed_emotion = smoothed_label

                text_y = y1 - 20 if y1 > 30 else y2 + 10
                cv2.rectangle(
                    frame,
                    (x1, text_y - 28),
                    (x1 + 220, text_y + 12),
                    (25, 25, 25),
                    -1,
                )
                emotion_text = displayed_emotion or label
                if show_confidence:
                    emotion_text = f"{emotion_text} ({confidence:.2f})"
                draw_text(frame, emotion_text, (x1 + 10, text_y), scale=0.9)
                last_seen_face = time.time()

        if current_face_label is None:
            if displayed_emotion is not None and (time.time() - last_seen_face) < 0.8:
                pass
            else:
                displayed_emotion = None
                label_history.clear()

        # Blend the BGRA overlay only when it exists and is currently enabled.
        if overlay is not None and overlay_enabled:
            overlay_x = frame.shape[1] - overlay_width - 10
            overlay_y = 10
            overlay_roi = frame[overlay_y : overlay_y + overlay_height, overlay_x : overlay_x + overlay_width]
            overlay_bgr = overlay[:, :, :3]
            overlay_alpha = overlay[:, :, 3:4] / 255.0
            frame[overlay_y : overlay_y + overlay_height, overlay_x : overlay_x + overlay_width] = (
                overlay_roi * (1.0 - overlay_alpha) + overlay_bgr * overlay_alpha
            ).astype(np.uint8)

        if overlay_enabled:
            elapsed = max(time.time() - start_time, 1e-6)
            fps = 0.9 * fps + 0.1 * (1.0 / elapsed) if fps else 1.0 / elapsed
            draw_text(frame, f"{fps:.1f} FPS", (18, 34), scale=0.42, color=(210, 220, 230), thickness=1)
            if show_help:
                help_text = "Q  Quit     O  Overlay     C  Confidence     R  Record     S  Snapshot     H  Hide help"
                bar_height = 44
                bar = frame.copy()
                cv2.rectangle(
                    bar,
                    (0, frame.shape[0] - bar_height),
                    (frame.shape[1], frame.shape[0]),
                    (18, 24, 34),
                    -1,
                )
                cv2.addWeighted(bar, 0.88, frame, 0.12, 0, frame)
                draw_text(
                    frame,
                    help_text,
                    (24, frame.shape[0] - 13),
                    scale=0.48,
                    color=(225, 232, 240),
                    thickness=1,
                )

        if recording and video_writer is not None:
            video_writer.write(frame)

        cv2.imshow(window_name, frame)

        # waitKey both updates the OpenCV window and reads the latest key press.
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("o"):
            overlay_enabled = not overlay_enabled
            print(f"Overlay enabled: {overlay_enabled}")
        elif key == ord("c"):
            show_confidence = not show_confidence
            print(f"Confidence display enabled: {show_confidence}")
        elif key == ord("h"):
            show_help = not show_help
        elif key == ord("s"):
            snapshot_path = config.snapshot_dir / f"snapshot_{int(time.time() * 1000)}.png"
            cv2.imwrite(str(snapshot_path), frame)
            print(f"Saved snapshot: {snapshot_path}")
        elif key == ord("r"):
            if not recording:
                # Use the actual frame dimensions because webcams and video
                # files may not provide the requested configuration size.
                output_path = config.recording_dir / f"recording_{int(time.time() * 1000)}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(str(output_path), fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                if not video_writer.isOpened():
                    raise RuntimeError(f"Could not open video writer for {output_path}")
                recording = True
                print(f"Started recording: {output_path}")
            else:
                recording = False
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                print("Stopped recording.")

    if video_writer is not None:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "display"], required=True, help="train/display")
    parser.add_argument("--overlay", help="Path to the overlay image", default=None)
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index to use for webcam capture")
    parser.add_argument("--input", help="Image or video path for no-camera input mode", default=None)
    parser.add_argument("--config", help="Path to a JSON configuration file", default=None)
    args = parser.parse_args()

    config = AppConfig.from_file(args.config)

    if args.mode == "train":
        train_model(config)
        return

    run_display(args.overlay, camera_index=args.camera_index, config=config, input_source=args.input)


if __name__ == "__main__":
    main()
