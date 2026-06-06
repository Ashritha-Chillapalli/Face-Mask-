import argparse
from pathlib import Path

import cv2
import numpy as np


AUTHORIZED_DIR = Path("authorized_faces")
FACE_SIZE = (200, 200)


def prepare_face(gray_frame, x, y, w, h):
    face = gray_frame[y : y + h, x : x + w]
    face = cv2.resize(face, FACE_SIZE)
    face = cv2.equalizeHist(face)
    return face


def train_authorized_recognizer(face_detector):
    faces = []
    labels = []
    label_names = {}
    next_label = 0

    AUTHORIZED_DIR.mkdir(exist_ok=True)
    for image_path in sorted(AUTHORIZED_DIR.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: could not read {image_path.name}; skipping.")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected_faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )
        if len(detected_faces) == 0:
            print(f"Warning: no face found in {image_path.name}; skipping.")
            continue

        x, y, w, h = max(detected_faces, key=lambda box: box[2] * box[3])
        faces.append(prepare_face(gray, x, y, w, h))
        labels.append(next_label)
        label_names[next_label] = image_path.stem.replace("_", " ").title()
        next_label += 1

    if not faces:
        return None, {}

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels, dtype=np.int32))
    return recognizer, label_names


def main() -> None:
    parser = argparse.ArgumentParser(description="Run intruder detection only.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument(
        "--face-threshold",
        type=float,
        default=70.0,
        help="LBPH recognition threshold. Lower is stricter.",
    )
    args = parser.parse_args()

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    recognizer, label_names = train_authorized_recognizer(face_detector)
    if recognizer is None:
        print("Warning: no authorized faces loaded. Add images to authorized_faces/.")

    video = cv2.VideoCapture(args.camera)
    if not video.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Running. Press q in the webcam window to quit.")

    while True:
        ok, frame = video.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            name = "Unknown"
            status = "Intruder"
            color = (0, 0, 255)

            if recognizer is not None:
                face = prepare_face(gray, x, y, w, h)
                label, confidence = recognizer.predict(face)
                if confidence <= args.face_threshold:
                    name = label_names.get(label, "Unknown")
                    status = "Authorized"
                    color = (0, 255, 0)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"{name} - {status}",
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        cv2.imshow("Intruder Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
