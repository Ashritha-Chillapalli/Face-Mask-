import argparse
from pathlib import Path

import cv2


AUTHORIZED_DIR = Path("authorized_faces")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture one clear authorized user image from the webcam."
    )
    parser.add_argument("name", help="Person name, for example harish")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Maximum seconds to wait for a face.",
    )
    args = parser.parse_args()

    AUTHORIZED_DIR.mkdir(exist_ok=True)
    output_path = AUTHORIZED_DIR / f"{args.name.lower().replace(' ', '_')}.jpg"

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    print("Look at the camera. Press s to save when your face has a green box.")
    print("The script will also auto-save the first clear detected face.")
    start_ticks = cv2.getTickCount()
    saved = False

    while True:
        ok, frame = camera.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(50, 50),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        elapsed = (cv2.getTickCount() - start_ticks) / cv2.getTickFrequency()
        remaining = max(0, int(args.timeout - elapsed))
        cv2.putText(
            frame,
            "Press s to save",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Timeout: {remaining}s",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Capture Authorized Face", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s") and len(faces) > 0:
            cv2.imwrite(str(output_path), frame)
            print(f"Saved authorized face: {output_path}")
            saved = True
            break

        if len(faces) > 0 and elapsed >= 2:
            cv2.imwrite(str(output_path), frame)
            print(f"Saved authorized face: {output_path}")
            saved = True
            break

        if elapsed >= args.timeout:
            if len(faces) == 0:
                print("No face detected. Try again with better lighting.")
            break

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    if not saved:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
