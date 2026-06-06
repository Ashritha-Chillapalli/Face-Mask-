import argparse
import csv
from datetime import datetime
from pathlib import Path
import time
import winsound

import cv2
import numpy as np

from detect_mask import load_mask_detector, predict_mask


AUTHORIZED_DIR = Path("authorized_faces")
LOG_DIR = Path("logs")
SCREENSHOT_DIR = Path("screenshots")
ATTENDANCE_FILE = LOG_DIR / "attendance.csv"
INTRUDER_FILE = LOG_DIR / "intruder_log.csv"
FACE_SIZE = (200, 200)


def ensure_runtime_dirs() -> None:
    AUTHORIZED_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    SCREENSHOT_DIR.mkdir(exist_ok=True)


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


def append_csv_once(path: Path, header, row) -> None:
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if new_file:
            writer.writerow(header)
        writer.writerow(row)


def log_attendance(name: str, status: str, mask_label: str) -> None:
    now = datetime.now()
    append_csv_once(
        ATTENDANCE_FILE,
        ["date", "time", "name", "status", "mask"],
        [now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), name, status, mask_label],
    )


def log_intruder(mask_label: str, screenshot_path: Path) -> None:
    now = datetime.now()
    append_csv_once(
        INTRUDER_FILE,
        ["date", "time", "mask", "screenshot"],
        [
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            mask_label,
            str(screenshot_path),
        ],
    )


def save_intruder_screenshot(frame) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SCREENSHOT_DIR / f"intruder_{timestamp}.jpg"
    cv2.imwrite(str(path), frame)
    return path


def beep_alert() -> None:
    try:
        winsound.Beep(1200, 400)
    except RuntimeError:
        pass


def recognize_face(gray_frame, x, y, w, h, recognizer, label_names, threshold: float):
    if recognizer is None:
        return "Unknown", "Intruder"

    face = prepare_face(gray_frame, x, y, w, h)
    label, confidence = recognizer.predict(face)

    if confidence > threshold:
        return "Unknown", "Intruder"

    return label_names.get(label, "Unknown"), "Authorized"


def draw_label(frame, x, y, w, h, name, status, mask_label, confidence):
    authorized = status == "Authorized"
    safe_mask = mask_label == "Mask"
    color = (0, 180, 0) if authorized and safe_mask else (0, 0, 255)

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    lines = [
        f"Name: {name}",
        f"Status: {status}",
        f"Mask: {mask_label} ({confidence * 100:.1f}%)",
    ]
    label_y = max(20, y - 55)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, label_y + index * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time face mask and intruder detection."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument(
        "--face-threshold",
        type=float,
        default=70.0,
        help="LBPH recognition threshold. Lower is stricter.",
    )
    parser.add_argument(
        "--alert-cooldown",
        type=int,
        default=10,
        help="Seconds between repeated intruder screenshots/alerts.",
    )
    parser.add_argument(
        "--log-cooldown",
        type=int,
        default=30,
        help="Seconds between repeated attendance logs for the same person.",
    )
    args = parser.parse_args()

    ensure_runtime_dirs()
    mask_model, mask_labels = load_mask_detector()
    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    recognizer, label_names = train_authorized_recognizer(face_detector)

    if recognizer is None:
        print("Warning: no authorized faces loaded. Add images to authorized_faces/.")

    video = cv2.VideoCapture(args.camera)
    if not video.isOpened():
        raise RuntimeError("Could not open webcam.")

    last_intruder_alert = 0.0
    last_attendance_log = {}

    print("Running. Press q in the webcam window to quit.")

    while True:
        ok, frame = video.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_crop = frame[y : y + h, x : x + w]
            if face_crop.size == 0:
                continue

            mask_label, confidence = predict_mask(face_crop, mask_model, mask_labels)
            name, status = recognize_face(
                gray,
                x,
                y,
                w,
                h,
                recognizer,
                label_names,
                args.face_threshold,
            )

            draw_label(frame, x, y, w, h, name, status, mask_label, confidence)

            now = time.time()
            log_key = f"{name}:{status}:{mask_label}"
            if now - last_attendance_log.get(log_key, 0.0) > args.log_cooldown:
                log_attendance(name, status, mask_label)
                last_attendance_log[log_key] = now

            if status == "Intruder" and now - last_intruder_alert > args.alert_cooldown:
                screenshot_path = save_intruder_screenshot(frame)
                log_intruder(mask_label, screenshot_path)
                beep_alert()
                last_intruder_alert = now

        cv2.imshow("Real-Time Face Mask and Intruder Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
