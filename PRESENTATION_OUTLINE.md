# Presentation Outline

## Slide 1: Title

Real-Time Face Mask and Intruder Detection

## Slide 2: Introduction

- AI-based security system
- Uses webcam or CCTV feed
- Detects masks and unauthorized persons

## Slide 3: Problem Statement

- Manual monitoring is slow
- Intruders may go unnoticed
- Mask compliance is hard to track manually

## Slide 4: Objectives

- Detect faces
- Classify mask or no mask
- Identify authorized people
- Detect intruders
- Save logs and screenshots

## Slide 5: Technologies

- Python
- OpenCV
- TensorFlow/Keras
- face_recognition
- NumPy
- scikit-learn

## Slide 6: Dataset

- Kaggle Face Mask Dataset
- Classes: with_mask and without_mask
- 80/20 train-validation split

## Slide 7: Architecture

```text
Webcam -> Face Detection -> Mask Detection
                         -> Face Recognition
                         -> Display, Logs, Alerts
```

## Slide 8: Model Training

- CNN model
- Image augmentation
- Accuracy and loss graph
- Classification report

## Slide 9: Realtime Detection

- Green box for safe authorized user
- Red box for intruder or no-mask case
- Displays name, status, and mask confidence

## Slide 10: Alerts and Logs

- Intruder screenshot
- Beep alert
- Attendance CSV
- Intruder CSV

## Slide 11: Results

- Show training graph
- Show webcam screenshot
- Show logs

## Slide 12: Limitations and Future Scope

- Lighting affects accuracy
- Masked faces are harder to recognize
- Future: YOLO, email alerts, dashboard, database

## Slide 13: Conclusion

The system provides a working real-time AI security solution using face detection, CNN mask classification, and intruder recognition.
