import argparse
import os
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.h5"
CASCADE_PATH = BASE_DIR / "haarcascade_frontalface_default.xml"
DEFAULT_OVERLAY_PATH = BASE_DIR / "overlay.png"
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "test"

IMAGE_SIZE = 48
EMOTION_LABELS = ["Angry", "Disgusted", "Fear", "Happy", "Neutral", "Sad", "Surprised"]
CONFIDENCE_THRESHOLD = 0.45
FACE_PADDING = 0.15

NUM_TRAIN = 28709
NUM_VAL = 7178
BATCH_SIZE = 64
NUM_EPOCHS = 50


def resolve_path(path_value):
    if path_value is None:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def build_model():
    model = Sequential(
        [
            Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
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
            Dense(7, activation="softmax"),
        ]
    )
    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.0001), metrics=["accuracy"])
    return model


def plot_model_history(model_history):
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
    fig.savefig(BASE_DIR / "plot.png")
    plt.show()


def train_model():
    if not TRAIN_DIR.exists() or not VAL_DIR.exists():
        raise FileNotFoundError(f"Training data directories not found under: {DATA_DIR}")

    train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        str(TRAIN_DIR),
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        class_mode="categorical",
    )

    validation_generator = val_datagen.flow_from_directory(
        str(VAL_DIR),
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        class_mode="categorical",
    )

    model = build_model()
    model_info = model.fit(
        train_generator,
        steps_per_epoch=NUM_TRAIN // BATCH_SIZE,
        epochs=NUM_EPOCHS,
        validation_data=validation_generator,
        validation_steps=NUM_VAL // BATCH_SIZE,
    )
    plot_model_history(model_info)
    model.save_weights(str(MODEL_PATH))
    print(f"Model saved to: {MODEL_PATH}")


def load_overlay(path_value):
    overlay_path = resolve_path(path_value)
    if overlay_path is None:
        overlay_path = DEFAULT_OVERLAY_PATH

    if not overlay_path.exists():
        raise FileNotFoundError(f"Overlay image not found: {overlay_path}")

    overlay = cv2.imread(str(overlay_path), cv2.IMREAD_UNCHANGED)
    if overlay is None:
        raise ValueError(f"Could not load overlay image: {overlay_path}")

    if overlay.shape[2] == 3:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

    return overlay


def run_display(overlay_value, camera_index=0):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = build_model()
    model.load_weights(str(MODEL_PATH))

    cv2.ocl.setUseOpenCL(False)

    cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
    if cascade.empty():
        raise FileNotFoundError(f"Face cascade file not found or invalid: {CASCADE_PATH}")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Warning: camera index {camera_index} could not be opened. Retrying with camera index 0...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam. Please ensure a camera is connected and available.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    overlay = load_overlay(overlay_value)
    overlay_width = 400
    overlay_height = int((overlay.shape[0] / overlay.shape[1]) * overlay_width)
    overlay = cv2.resize(overlay, (overlay_width, overlay_height), interpolation=cv2.INTER_AREA)

    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Video", 100, 100)

    displayed_emotion = None
    label_history = deque(maxlen=5)
    last_seen_face = time.time()

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam. Retrying...")
            time.sleep(0.5)
            if not cap.isOpened():
                break
            continue

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
            pad_x = int(w * FACE_PADDING)
            pad_y = int(h * FACE_PADDING)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(frame.shape[1], x + w + pad_x)
            y2 = min(frame.shape[0], y + h + pad_y)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (192, 192, 192), 2)

            roi_gray = gray[y1:y2, x1:x2]
            roi_resized = cv2.resize(roi_gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            roi_normalized = roi_resized.astype(np.float32) / 255.0
            roi_batch = roi_normalized[:, :, np.newaxis][np.newaxis, ...]

            prediction = model(roi_batch, training=False).numpy()[0]
            max_index = int(np.argmax(prediction))
            confidence = float(prediction[max_index])
            label = EMOTION_LABELS[max_index]

            if confidence >= CONFIDENCE_THRESHOLD:
                current_face_label = label
                label_history.append(label)
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
                cv2.putText(frame, displayed_emotion or label, (x1 + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                last_seen_face = time.time()

        if current_face_label is None:
            if displayed_emotion is not None and (time.time() - last_seen_face) < 0.8:
                pass
            else:
                displayed_emotion = None
                label_history.clear()

        overlay_x = frame.shape[1] - overlay_width - 10
        overlay_y = 10
        overlay_roi = frame[overlay_y : overlay_y + overlay_height, overlay_x : overlay_x + overlay_width]
        overlay_bgr = overlay[:, :, :3]
        overlay_alpha = overlay[:, :, 3:4] / 255.0

        frame[overlay_y : overlay_y + overlay_height, overlay_x : overlay_x + overlay_width] = (
            overlay_roi * (1.0 - overlay_alpha) + overlay_bgr * overlay_alpha
        ).astype(np.uint8)

        footer_text = "By Lindner&Sumann"
        footer_y = frame.shape[0] - 20
        cv2.putText(frame, footer_text, (20, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        _ = time.time() - start_time

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "display"], required=True, help="train/display")
    parser.add_argument("--overlay", help="Path to the overlay image", default=None)
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index to use for webcam capture")
    args = parser.parse_args()

    if args.mode == "train":
        train_model()
        return

    run_display(args.overlay, camera_index=args.camera_index)


if __name__ == "__main__":
    main()
