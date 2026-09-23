import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode", help="train/display")
ap.add_argument("--overlay", help="Path to the overlay image", default='src/overlay.png')  # Default path
args = ap.parse_args()
mode = args.mode
overlay_path = args.overlay


# Function to plot model history
def plot_model_history(model_history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(range(1, len(model_history.history['accuracy']) + 1), model_history.history['accuracy'])
    axs[0].plot(range(1, len(model_history.history['val_accuracy']) + 1), model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['train', 'val'], loc='best')
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    fig.savefig('plot.png')
    plt.show()


# Define data generators
train_dir = 'data/train'
val_dir = 'data/test'
num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = 50

train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical'
)

# Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# If you want to train the same model or try other models
if mode == "train":
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001, decay=1e-6),
                  metrics=['accuracy'])
    model_info = model.fit(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_val // batch_size
    )
    plot_model_history(model_info)
    model.save_weights('model.h5')

# Emotions will be displayed on your face from the webcam feed
elif mode == "display":
    model.load_weights('model.h5')
    cv2.ocl.setUseOpenCL(False)

    emotion_dict = {0: "Wuetend", 1: "Angewidert", 2: "Neutral", 3: "Gluecklich", 4: "Neutral", 5: "Traurig",
                    6: "Ueberrascht"}

    cap = cv2.VideoCapture(0)

    # Set resolution to improve performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Reduce width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Reduce height

    overlay_image = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    if overlay_image is None:
        print("Error: Could not load overlay image. Check the file path and file integrity.")
        exit()

    if overlay_image.shape[2] == 3:
        overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2BGRA)

    overlay_width = 400  # Increased width of logo
    overlay_height = int((overlay_image.shape[0] / overlay_image.shape[1]) * overlay_width)
    overlay_image = cv2.resize(overlay_image, (overlay_width, overlay_height))

    cv2.namedWindow("Video", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        start_time = time.time()  # Start time for FPS calculation
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml').detectMultiScale(gray, scaleFactor=1.3,
                                                                                              minNeighbors=5)

        for (x, y, w, h) in faces:
            # Softer gray border around the face
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (192, 192, 192), 2)  # Light gray (192, 192, 192)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))

            text_y = y - 60 if y - 60 > 0 else y + h + 20
            if maxindex in emotion_dict:
                # Background for the emotion text with slightly transparent dark gray
                text = emotion_dict[maxindex]
                (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)

                # Validate text dimensions
                if text_w is None or text_h is None:
                    print("Error: Text width or height is None.")
                    continue

                overlay_color = (50, 50, 50)
                alpha = 0.6  # Transparency value (between 0 and 1)

                # Ensure valid indices for sub_img
                sub_img_y_start = max(0, text_y - text_h - 5)
                sub_img_y_end = min(frame.shape[0], text_y + 5)
                sub_img_x_start = x + 20
                sub_img_x_end = x + 20 + text_w

                # Ensure we are within bounds
                sub_img = frame[sub_img_y_start:sub_img_y_end, sub_img_x_start:sub_img_x_end]
                rect = np.ones(sub_img.shape, dtype=np.uint8) * np.array(overlay_color, dtype=np.uint8)
                blended = cv2.addWeighted(rect, alpha, sub_img, 1 - alpha, 0)
                frame[sub_img_y_start:sub_img_y_end, sub_img_x_start:sub_img_x_end] = blended

                # Emotion text in white with subtle shadow for better readability
                cv2.putText(frame, text, (x + 20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2,
                            cv2.LINE_AA)

        # Overlay image without black background
        overlay_x = frame.shape[1] - overlay_width - 10
        overlay_y = 10

        for c in range(0, 3):
            frame[overlay_y:overlay_y + overlay_height, overlay_x:overlay_x + overlay_width, c] = (
                    overlay_image[:, :, c] * (overlay_image[:, :, 3] / 255.0) +
                    frame[overlay_y:overlay_y + overlay_height, overlay_x:overlay_x + overlay_width, c] * (
                            1.0 - overlay_image[:, :, 3] / 255.0)
            )

        # Add footer text in the bottom left corner
        footer_text = "By Lindner&Sumann"
        footer_text_y = frame.shape[0] - 20  # Position slightly above the bottom
        (footer_text_w, footer_text_h), footer_baseline = cv2.getTextSize(footer_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

        # Create background for footer text
        footer_sub_img_y_start = max(0, footer_text_y - footer_text_h - 5)
        footer_sub_img_y_end = min(frame.shape[0], footer_text_y + 5)
        footer_sub_img_x_start = 20
        footer_sub_img_x_end = 20 + footer_text_w

        # Ensure we are within bounds
        footer_sub_img = frame[footer_sub_img_y_start:footer_sub_img_y_end, footer_sub_img_x_start:footer_sub_img_x_end]
        footer_rect = np.ones(footer_sub_img.shape, dtype=np.uint8) * np.array((0, 0, 0), dtype=np.uint8)  # Black background
        alpha = 0.6  # Transparency value
        footer_blended = cv2.addWeighted(footer_rect, alpha, footer_sub_img, 1 - alpha, 0)
        frame[footer_sub_img_y_start:footer_sub_img_y_end, footer_sub_img_x_start:footer_sub_img_x_end] = footer_blended

        # Add footer text in white
        cv2.putText(frame, footer_text, (20, footer_text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow("Video", frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()