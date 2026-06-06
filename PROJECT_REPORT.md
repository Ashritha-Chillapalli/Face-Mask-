# Real-Time Face Mask and Intruder Detection

## Abstract

This project implements a real-time AI security system using a webcam or CCTV feed. It detects human faces, classifies whether each detected person is wearing a mask, and identifies whether the person is authorized or an intruder. The system also records attendance logs and saves screenshots when an intruder is detected.

## Problem Statement

Manual security monitoring is time-consuming and error-prone. A smart automated system can help identify unknown persons and check mask compliance in real time, making it useful for campuses, offices, labs, and restricted areas.

## Objectives

- Detect faces from a live camera feed.
- Train a CNN model to classify mask and no-mask faces.
- Recognize authorized users from stored face images.
- Detect unknown users as intruders.
- Save intruder screenshots and log events.
- Maintain attendance records with mask status.

## Technologies Used

- Python
- OpenCV
- TensorFlow/Keras
- OpenCV LBPH face recognition
- NumPy
- scikit-learn
- Matplotlib

## Dataset

Dataset used: Face Mask Dataset from Kaggle  
URL: https://kaggle.com/datasets/omkargurav/face-mask-dataset

The dataset is divided into:

- `with_mask`
- `without_mask`

The training script uses an 80/20 train-validation split.

## System Architecture

```text
Webcam Feed
    |
    v
OpenCV Face Detection
    |
    +------------------------+
    |                        |
    v                        v
CNN Mask Classifier     Face Recognition
    |                        |
    v                        v
Mask / No Mask          Authorized / Intruder
    |                        |
    +-----------+------------+
                |
                v
      Display + Logs + Alerts
```

## Methodology

1. Images are loaded from the dataset folders.
2. A CNN model is trained with data augmentation.
3. The trained model is saved in the `models` folder.
4. Authorized user face images are loaded from `authorized_faces`.
5. The webcam feed is processed frame by frame.
6. Each detected face is passed to the mask model.
7. The same face is compared against the trained LBPH authorized-face recognizer.
8. Intruders trigger a beep alert, screenshot, and CSV log.

## CNN Model

The mask model uses convolution, batch normalization, max pooling, dense layers, and dropout. The final layer uses softmax activation for two-class classification.

## Outputs

- Realtime webcam display
- Mask / No Mask classification
- Authorized / Intruder classification
- `logs/attendance.csv`
- `logs/intruder_log.csv`
- `screenshots/intruder_*.jpg`
- `reports/training_history.png`
- `reports/mask_model_report.txt`

## Advantages

- Works in real time.
- Uses a custom-trained CNN model.
- Stores useful evidence for intruder events.
- Easy to extend with email, SMS, or database storage.

## Limitations

- Face recognition accuracy depends on image quality and lighting.
- Masked faces may be harder to recognize accurately.
- Webcam quality affects detection performance.
- LBPH recognition works best when authorized images are clear and front-facing.

## Future Enhancements

- Add YOLO-based face detection for improved accuracy.
- Add email or SMS alerts.
- Add a web dashboard.
- Store logs in a database.
- Add multiple camera support.

## Conclusion

The project successfully combines computer vision and deep learning to create a functional real-time security system. It can detect masks, identify authorized users, flag intruders, and generate logs suitable for academic demonstration and practical extension.
