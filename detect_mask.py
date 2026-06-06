import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model


IMG_SIZE = 224
MODEL_PATH = Path("models/mask_detector.keras")
LABELS_PATH = Path("models/mask_labels.json")


def load_mask_detector(model_path: Path = MODEL_PATH, labels_path: Path = LABELS_PATH):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Mask model not found: {model_path}. Train it first with python train_mask_detector.py"
        )
    model = load_model(model_path)

    if labels_path.exists():
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
    else:
        labels = ["with_mask", "without_mask"]

    return model, labels


def normalize_mask_label(label: str) -> str:
    clean = label.lower().replace(" ", "_").replace("-", "_")
    if "without" in clean or "no_mask" in clean or clean == "nomask":
        return "No Mask"
    return "Mask"


def predict_mask(face_bgr, model, labels):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
    face_rgb = face_rgb.astype("float32") / 255.0
    face_rgb = np.expand_dims(face_rgb, axis=0)

    probabilities = model.predict(face_rgb, verbose=0)[0]
    class_index = int(np.argmax(probabilities))
    label = normalize_mask_label(labels[class_index])
    confidence = float(probabilities[class_index])
    return label, confidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Run webcam mask detection only.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    args = parser.parse_args()

    model, labels = load_mask_detector()
    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    video = cv2.VideoCapture(args.camera)
    if not video.isOpened():
        raise RuntimeError("Could not open webcam.")

    while True:
        ok, frame = video.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = frame[y : y + h, x : x + w]
            if face.size == 0:
                continue

            mask_label, confidence = predict_mask(face, model, labels)
            color = (0, 255, 0) if mask_label == "Mask" else (0, 0, 255)
            text = f"{mask_label}: {confidence * 100:.1f}%"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                text,
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        cv2.imshow("Mask Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
