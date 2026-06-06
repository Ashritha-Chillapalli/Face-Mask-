from pathlib import Path
import sys

import cv2
import tensorflow as tf


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(
        1
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def status(ok: bool) -> str:
    return "PASS" if ok else "MISSING"


def check_camera() -> bool:
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        return False
    ok, _ = camera.read()
    camera.release()
    return bool(ok)


def check_authorized_faces(face_detector) -> tuple[int, int]:
    folder = Path("authorized_faces")
    total = 0
    usable = 0

    for path in folder.iterdir() if folder.exists() else []:
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        total += 1
        image = cv2.imread(str(path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) > 0:
            usable += 1

    return total, usable


def main() -> int:
    with_mask = count_images(Path("dataset/with_mask"))
    without_mask = count_images(Path("dataset/without_mask"))
    model_path = Path("models/mask_detector.keras")
    labels_path = Path("models/mask_labels.json")

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    authorized_total, authorized_usable = check_authorized_faces(face_detector)
    camera_ok = check_camera()
    has_lbph = hasattr(cv2, "face")
    gpu_devices = tf.config.list_physical_devices("GPU")

    print("\nProject Verification")
    print("====================")
    print(f"Python: {sys.version.split()[0]}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"OpenCV LBPH available: {status(has_lbph)}")
    print(f"TensorFlow GPU devices: {gpu_devices if gpu_devices else 'None detected'}")
    print(f"Webcam frame capture: {status(camera_ok)}")
    print(f"Dataset with_mask images: {with_mask}")
    print(f"Dataset without_mask images: {without_mask}")
    print(f"Authorized face images: {authorized_total}")
    print(f"Authorized usable faces: {authorized_usable}")
    print(f"Mask model file: {status(model_path.exists())} ({model_path})")
    print(f"Mask labels file: {status(labels_path.exists())} ({labels_path})")

    ready_to_train = with_mask > 0 and without_mask > 0
    ready_to_run = (
        ready_to_train
        and model_path.exists()
        and labels_path.exists()
        and authorized_usable > 0
        and camera_ok
        and has_lbph
    )

    print("\nReadiness")
    print("---------")
    print(f"Ready to train mask model: {status(ready_to_train)}")
    print(f"Ready to run full realtime app: {status(ready_to_run)}")

    if not ready_to_train:
        print("\nAction needed: add Kaggle images into dataset/with_mask and dataset/without_mask.")
    if authorized_usable == 0:
        print("Action needed: add at least one clear front-facing image into authorized_faces/.")
    if ready_to_train and not model_path.exists():
        print("Action needed: run python train_mask_detector.py --epochs 15 --batch-size 32.")
    if not camera_ok:
        print("Action needed: check webcam permission or close apps already using the camera.")

    if not gpu_devices:
        print(
            "\nGPU note: native Windows TensorFlow 2.11+ does not use NVIDIA GPU. "
            "The app still works on CPU. For GPU training, use WSL2 + CUDA or a DirectML setup."
        )

    return 0 if ready_to_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
