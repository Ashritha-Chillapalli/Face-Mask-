# Real-Time Face Mask and Intruder Detection

This project is an AI-based security system that uses a webcam feed to detect faces, classify mask usage, and identify authorized people or intruders in real time.

## Features

- Face detection with OpenCV Haar Cascade
- Mask detection using a CNN trained with TensorFlow/Keras
- Authorized-person recognition using OpenCV LBPH face recognition
- Intruder detection for unknown faces
- Intruder screenshot capture
- Windows beep alert for intruders
- Attendance and intruder CSV logs
- Training graph and model evaluation report

## Project Structure

```text
authorized_faces/
dataset/
  with_mask/
  without_mask/
logs/
models/
reports/
screenshots/
detect_mask.py
intruder_detection.py
main.py
train_mask_detector.py
requirements.txt
```

## Setup

Create and activate the virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

The project uses `opencv-contrib-python` because it includes OpenCV's LBPH face recognizer.

## Dataset

Download and extract the Kaggle dataset:

https://kaggle.com/datasets/omkargurav/face-mask-dataset

This project can also download it automatically:

```powershell
python -c "import kagglehub; print(kagglehub.dataset_download('omkargurav/face-mask-dataset'))"
```

The folders must look like this:

```text
dataset/
  with_mask/
    image1.jpg
    image2.jpg
  without_mask/
    image1.jpg
    image2.jpg
```

## Authorized Faces

Add clear front-facing images of authorized people:

```text
authorized_faces/
  harish.jpg
  admin.jpg
```

The filename becomes the display name in the webcam window.

You can capture your authorized image directly from the webcam:

```powershell
python capture_authorized_face.py harish
```

## Train the Mask Model

```powershell
python train_mask_detector.py --epochs 15 --batch-size 32
```

Outputs:

- `models/mask_detector.keras`
- `models/mask_labels.json`
- `reports/training_history.png`
- `reports/mask_model_report.txt`

## Run the Full Project

```powershell
python main.py
```

Press `q` in the webcam window to quit.

The app displays:

- Name
- Authorized / Intruder status
- Mask / No Mask result

Runtime outputs:

- `logs/attendance.csv`
- `logs/intruder_log.csv`
- `screenshots/intruder_*.jpg`

## Optional Test Modes

Verify all setup items:

```powershell
python verify_project.py
```

Run mask detection only:

```powershell
python detect_mask.py
```

Run intruder detection only:

```powershell
python intruder_detection.py
```

## GPU / CPU Training Note

Check available TensorFlow devices with:

```powershell
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Native Windows TensorFlow 2.11+ usually prints an empty GPU list even on NVIDIA systems. That is normal. The project still trains and runs on CPU. For GPU acceleration on Windows, use WSL2 with NVIDIA CUDA support or a DirectML-compatible TensorFlow setup.
