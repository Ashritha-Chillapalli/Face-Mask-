import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMG_SIZE = 224
DEFAULT_DATASET_DIR = Path("dataset")
DEFAULT_MODEL_PATH = Path("models/mask_detector.keras")
DEFAULT_LABELS_PATH = Path("models/mask_labels.json")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_model(num_classes: int) -> Sequential:
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(64, (3, 3), activation="relu"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(128, (3, 3), activation="relu"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(128, (3, 3), activation="relu"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),

            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def count_images(folder: Path) -> int:
    return sum(
        1
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_dataset(dataset_dir: Path) -> None:
    with_mask = dataset_dir / "with_mask"
    without_mask = dataset_dir / "without_mask"
    if not with_mask.exists() or not without_mask.exists():
        raise FileNotFoundError(
            "Dataset must contain dataset/with_mask and dataset/without_mask folders."
        )

    with_count = count_images(with_mask)
    without_count = count_images(without_mask)
    if with_count == 0 or without_count == 0:
        raise ValueError(
            "Dataset folders are empty. Add Kaggle images before training. "
            f"Found with_mask={with_count}, without_mask={without_count}."
        )


def plot_training(history, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_report(model, validation_generator, labels, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    validation_generator.reset()
    probabilities = model.predict(validation_generator, verbose=1)
    predictions = np.argmax(probabilities, axis=1)
    actual = validation_generator.classes

    text_report = classification_report(
        actual,
        predictions,
        target_names=labels,
        zero_division=0,
    )
    matrix = confusion_matrix(actual, predictions)

    report_path.write_text(
        "Classification Report\n"
        "=====================\n\n"
        f"{text_report}\n\n"
        "Confusion Matrix\n"
        "================\n\n"
        f"{matrix}\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the face mask detector.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_DIR), help="Dataset folder.")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_dir}. Expected dataset/with_mask and dataset/without_mask."
        )
    validate_dataset(dataset_dir)

    DEFAULT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2,
    )

    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=args.batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    validation_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=args.batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    labels_by_index = {
        index: label for label, index in train_generator.class_indices.items()
    }
    labels = [labels_by_index[index] for index in sorted(labels_by_index)]
    DEFAULT_LABELS_PATH.write_text(json.dumps(labels, indent=2), encoding="utf-8")

    model = build_model(num_classes=len(labels))
    callbacks = [
        ModelCheckpoint(
            DEFAULT_MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model.save(DEFAULT_MODEL_PATH)
    plot_training(history, Path("reports/training_history.png"))
    save_report(model, validation_generator, labels, Path("reports/mask_model_report.txt"))

    print(f"\nTraining complete.")
    print(f"Model saved to: {DEFAULT_MODEL_PATH}")
    print(f"Labels saved to: {DEFAULT_LABELS_PATH}")
    print("Reports saved in: reports/")


if __name__ == "__main__":
    main()
