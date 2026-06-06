# Client Guide: Real-Time Face Mask and Intruder Detection

## 1. Project Overview

This project is an AI-based security monitoring system. It uses a webcam or CCTV-style camera feed to detect people in real time, check whether they are wearing a mask, and identify whether they are an authorized person or an intruder.

The project is suitable for a college/demo submission because it includes:

- A trained CNN mask detection model
- Real-time webcam detection
- Authorized person recognition
- Intruder detection
- Attendance logging
- Intruder screenshot capture
- Model training report and graph

## 2. Requirement Coverage

| Requirement | Status | Implementation |
| --- | --- | --- |
| Face Detection | Completed | OpenCV Haar Cascade detects faces from webcam frames |
| Mask Detection | Completed | TensorFlow/Keras CNN classifies Mask or No Mask |
| Dataset Training | Completed | Kaggle face mask dataset is used for training |
| Intruder Detection | Completed | OpenCV LBPH recognizes authorized users and marks unknown faces as intruders |
| Realtime Webcam Feed | Completed | `main.py` processes webcam frames live |
| Logs | Completed | Attendance and intruder logs are stored as CSV files |
| Intruder Screenshot | Completed | Intruder frames are saved in `screenshots/` |
| Verification Script | Completed | `verify_project.py` checks readiness before demo |

## 3. How the System Works

```text
Webcam Frame
    |
    v
OpenCV Face Detection
    |
    +--------------------------+
    |                          |
    v                          v
Mask Detection CNN       Authorized Face Recognition
    |                          |
    v                          v
Mask / No Mask           Authorized / Intruder
    |                          |
    +------------+-------------+
                 |
                 v
        Display Result + Save Logs
```

Step-by-step flow:

1. The webcam captures a video frame.
2. OpenCV detects faces in the frame.
3. Each detected face is sent to the trained TensorFlow CNN model.
4. The CNN predicts whether the face belongs to `Mask` or `No Mask`.
5. The same face is processed by OpenCV LBPH face recognition.
6. If the face matches an image in `authorized_faces/`, the person is marked as Authorized.
7. If the face does not match, the person is marked as Intruder.
8. The result is displayed on the webcam window.
9. Attendance and intruder events are saved in CSV logs.
10. Intruder screenshots are saved automatically.

## 4. Main Files

| File | Purpose |
| --- | --- |
| `main.py` | Runs the full real-time system |
| `train_mask_detector.py` | Trains the CNN mask detection model |
| `detect_mask.py` | Runs only mask detection |
| `intruder_detection.py` | Runs only authorized/intruder detection |
| `capture_authorized_face.py` | Captures an authorized user image from the webcam |
| `verify_project.py` | Checks whether the project is ready to run |
| `PROJECT_REPORT.md` | Academic project report |
| `PRESENTATION_OUTLINE.md` | PPT slide content outline |
| `README.md` | Setup and run instructions |

## 5. Dataset

Dataset used:

https://kaggle.com/datasets/omkargurav/face-mask-dataset

Current dataset count:

- `dataset/with_mask`: 3725 images
- `dataset/without_mask`: 3828 images

The training script uses an 80/20 train-validation split.

## 6. Trained Model

The trained mask detection model is saved here:

```text
models/mask_detector.keras
```

The model labels are saved here:

```text
models/mask_labels.json
```

Current validation result:

```text
Accuracy: 92%
with_mask precision: 0.92
without_mask precision: 0.92
```

Detailed model evaluation is available in:

```text
reports/mask_model_report.txt
reports/training_history.png
```

## 7. How to Run the Project

Open PowerShell in the project folder and activate the environment:

```powershell
venv\Scripts\activate
```

Check that everything is ready:

```powershell
python verify_project.py
```

Run the full project:

```powershell
python main.py
```

Press `q` in the webcam window to close the application.

## 8. Output Files

The system creates these output files while running:

```text
logs/attendance.csv
logs/intruder_log.csv
screenshots/intruder_*.jpg
```

Attendance log example:

```text
date,time,name,status,mask
2026-06-05,12:17:20,Harish,Authorized,No Mask
2026-06-05,12:17:44,Harish,Authorized,Mask
```

Intruder log example:

```text
date,time,mask,screenshot
2026-06-05,12:17:25,No Mask,screenshots\intruder_20260605_121725.jpg
```

## 9. Verification Status

The project was verified with:

```powershell
python verify_project.py
```

Current result:

```text
Ready to train mask model: PASS
Ready to run full realtime app: PASS
```

Confirmed items:

- Python environment works
- OpenCV works
- TensorFlow works
- Webcam works
- Dataset exists
- Trained model exists
- Authorized face exists
- Full realtime app runs

## 10. NVIDIA GPU Note

The laptop has an NVIDIA GPU, but native Windows TensorFlow 2.11+ does not use NVIDIA GPU directly. TensorFlow reports no GPU devices in this environment.

This does not stop the project from working. The model was trained successfully on CPU, and the realtime application runs correctly.

For GPU acceleration, use one of these setups:

- WSL2 with NVIDIA CUDA support
- TensorFlow DirectML-compatible setup

## 11. Limitations

- Accuracy depends on lighting and camera quality.
- Authorized face recognition works best with clear front-facing images.
- Masked faces can be harder to identify.
- Haar Cascade face detection is lightweight but less advanced than YOLO.

## 12. Final Submission Package

The final project ZIP is:

```text
FaceMaskIntruderDetection_Submission.zip
```

It includes the source code, dataset, trained model, reports, logs, screenshots, and documentation required for submission.
