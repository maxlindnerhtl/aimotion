# AI-Motion

<img src="assets/demo.gif" width="600" alt="AI-Motion Demo">

AI-Motion is a Python-based computer vision application that detects faces and estimates visible facial expressions in real time. It supports webcam input as well as image and video files.

The application classifies expressions into seven categories:

- Angry
- Disgusted
- Fear
- Happy
- Neutral
- Sad
- Surprised

> **Important:** The model estimates facial expressions from visual patterns. It does not reliably determine a person's actual emotional state and should not be used for medical, psychological, employment, or other high-stakes decisions.

## Features

- Real-time expression recognition through a webcam
- Image and video file input without a camera
- Configurable runtime settings through `src/config.json`
- Optional visual overlay
- Optional confidence display
- Snapshot capture
- Video recording
- Model training support
- Training metadata for reproducibility
- Smoke tests and GitHub Actions CI

## How It Works

The application uses:

- **OpenCV** for image capture, face detection, and display
- **TensorFlow/Keras** for expression classification
- A convolutional neural network trained on FER-2013-style facial expression data
- A Haar cascade file for face detection

The processing pipeline is:

1. Capture a frame from a webcam, image, or video.
2. Detect one or more faces.
3. Crop and resize each face to 48 × 48 pixels.
4. Pass the face image to the neural network.
5. Select the expression with the highest prediction score.
6. Display the result in the application window.

## Requirements

The documented setup targets:

- Windows 10 or Windows 11
- Python 3.11
- Git
- A webcam, or an image/video file for camera-free usage
- Internet access for installing Python dependencies

Python 3.11 is recommended because it is the version used with the project's TensorFlow setup.

## Quick Start on Windows

### 1. Clone the repository

Open PowerShell and run:

```powershell
git clone https://github.com/maxlindnerhtl/aimotion.git
cd aimotion
```

### 2. Verify Python

Check whether Python 3.11 is available:

```powershell
py -3.11 --version
```

### 3. Create a virtual environment

Create an isolated Python environment in the project directory:

```powershell
py -3.11 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

After activation, `(.venv)` should appear at the beginning of the PowerShell prompt.

### 4. Install dependencies

With the virtual environment activated, install the required packages:

```powershell
pip install -r src\requirements.txt
```

### 5. Start the application

From the project root, start webcam mode with the default configuration:

```powershell
python .\src\emotions.py --mode display --config .\src\config.json
```

Alternatively, run the application from the `src` directory:

```powershell
cd src
python emotions.py --mode display --config config.json
```

The application window should open and display the camera feed. Detected faces will be marked and classified expressions will be shown above them.

## Keyboard Controls

The controls are available while the application window is focused. A small help bar is displayed along the bottom of the fullscreen window by default.

| Key | Action |
| --- | --- |
| `Q` | Quit the application |
| `O` | Toggle the complete UI overlay, including the decorative image, FPS indicator, and help bar |
| `C` | Toggle confidence values |
| `R` | Start or stop video recording |
| `S` | Save the current frame as a snapshot |
| `H` | Show or hide the bottom help bar |

Recordings are stored in `src/recordings` by default. Snapshots are stored in `src/snapshots`.

Face rectangles and expression labels remain visible when the UI overlay is disabled because they are the primary recognition output.

## Running Without a Webcam

You can process an image or video file by passing its path through `--input`:

```powershell
python .\src\emotions.py --mode display --input .\example.mp4
```

For an image:

```powershell
python .\src\emotions.py --mode display --input .\example.png
```

When processing a video file, the application exits after the final frame. Press `Q` to stop playback early.

## Configuration

Runtime settings are stored in `src/config.json`. This allows common settings to be changed without modifying Python code.

Example:

```json
{
  "image_size": 48,
  "camera_width": 1280,
  "camera_height": 720,
  "overlay_enabled": true,
  "show_confidence": true,
  "confidence_threshold": 0.45,
  "face_padding": 0.15,
  "num_train": 28709,
  "num_val": 7178,
  "batch_size": 64,
  "num_epochs": 50,
  "recording_dir": "recordings",
  "snapshot_dir": "snapshots"
}
```

Important settings:

- `camera_width` and `camera_height`: Requested webcam resolution. The application uses this fixed resolution during a session; change the values in `config.json` before starting the program.
- `overlay_enabled`: Whether the overlay is enabled when the application starts.
- `show_confidence`: Whether prediction confidence values are shown initially.
- `confidence_threshold`: Minimum confidence required before an expression is displayed.
- `face_padding`: Additional area included around each detected face.
- `recording_dir`: Directory for recorded videos.
- `snapshot_dir`: Directory for saved snapshots.
- `image_size`: Neural network input size. This should remain `48` unless the model is retrained with a different input size.

You can also use a different configuration file:

```powershell
python .\src\emotions.py --mode display --config .\my-config.json
```

## Training the Model

Training data must be available under the expected directory structure in `src/data`:

```text
src/data/
├── train/
│   ├── Angry/
│   ├── Disgusted/
│   └── ...
└── test/
    ├── Angry/
    ├── Disgusted/
    └── ...
```

Start training with:

```powershell
python .\src\emotions.py --mode train --config .\src\config.json
```

After training:

- Model weights are saved to `src/model.h5`.
- A training plot is saved to `src/plot.png`.
- Training metadata is saved to `src/artifacts/training_manifest.json`.

Training can take a significant amount of time and may require substantial memory. For normal usage, the included model can be used directly without retraining.

## Project Structure

```text
aimotion/
├── .github/
│   └── workflows/
│       └── ci.yml
├── assets/
│   └── demo.gif
├── README.md
├── pytest.ini
├── src/
│   ├── artifacts/
│   ├── config.json
│   ├── config.py
│   ├── data/
│   ├── emotions.py
│   ├── haarcascade_frontalface_default.xml
│   ├── model.h5
│   ├── model_utils.py
│   ├── overlay.png
│   ├── reproducibility.py
│   └── requirements.txt
├── tests/
│   ├── test_reproducibility.py
│   └── test_smoke.py
└── .venv/
```

The `.venv` directory is generated locally and is not intended to be committed to version control.

## Testing

Run the smoke tests from the project root:

```powershell
python -m pytest tests/test_smoke.py -q
```

Run all current tests:

```powershell
python -m pytest tests/test_smoke.py tests/test_reproducibility.py -q
```

GitHub Actions runs the test suite automatically for pull requests and for pushes to the `main` or `master` branches.

## Reproducibility

Each training run creates `src/artifacts/training_manifest.json`. The manifest records:

- Training timestamp
- Model and training configuration
- Dataset paths
- Model path
- Training and validation history
- Git commit, when available

This makes it easier to understand how a specific model was produced and to compare future training runs.

## Troubleshooting

### Python 3.11 cannot be found

Run:

```powershell
py -3.11 --version
```

If the command fails, install Python 3.11 and ensure that the Python Launcher is available.

### `No module named ...`

Make sure the virtual environment is active. The prompt should start with `(.venv)`. Then reinstall the dependencies:

```powershell
pip install -r src\requirements.txt
```

### The webcam does not open

Check that:

- The webcam is connected.
- No other application is using the webcam.
- Windows camera permissions allow access.
- The correct camera index is being used.

You can try another camera index:

```powershell
python .\src\emotions.py --mode display --camera-index 1
```

### `model.h5` is missing

The application requires trained model weights. Either restore `src/model.h5` or train a model:

```powershell
python .\src\emotions.py --mode train --config .\src\config.json
```

### The overlay cannot be loaded

Check that `src/overlay.png` exists. You can also disable the overlay at startup by setting `"overlay_enabled": false` in `src/config.json`, or press `O` while the application is running.

### PowerShell does not allow script activation

If PowerShell blocks activation scripts, run PowerShell as your user and execute:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate the environment again:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Optional: Build a Windows Executable

PyInstaller can package the application as a Windows executable. This is optional; the Python setup above is the recommended development workflow.

Install PyInstaller:

```powershell
pip install pyinstaller
```

From the `src` directory, run:

```powershell
cd src
pyinstaller --onefile --add-data "haarcascade_frontalface_default.xml;." --add-data "overlay.png;." emotions.py
```

The executable is created in the `dist` directory. Packaging TensorFlow applications may require additional PyInstaller configuration and should be tested on the target machine.

## Limitations

Expression recognition is sensitive to lighting, camera quality, pose, occlusion, and dataset bias. A confidence score is not a guarantee that the prediction is correct. Treat the output as an experimental estimate rather than an objective measurement of a person's feelings.
