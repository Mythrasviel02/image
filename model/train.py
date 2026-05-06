import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


# Map raw dataset folder names to the final 5-category requirement.
CLASS_MAPPING = {
    "plastic": "plastic",
    "paper": "paper",
    "cardboard": "paper",
    "glass": "glass",
    "metal": "metal",
    "trash": "organic_waste",
}

SEED = 42
IMG_SIZE = (224, 224)


def parse_args() -> argparse.Namespace:
    # CLI args make it easy to run the same script locally or in cloud training jobs.
    parser = argparse.ArgumentParser(description="Train garbage classifier with MobileNetV2")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="Garbage classification/Garbage classification",
        help="Path to raw dataset folder (contains class subfolders)",
    )
    parser.add_argument(
        "--prepared-dir",
        type=str,
        default="artifacts/splits",
        help="Output path for 70/15/15 prepared dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Directory to save model and evaluation outputs",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--fine-tune-layers", type=int, default=200)
    parser.add_argument("--fine-tune-lr-mult", type=float, default=0.05)
    parser.add_argument(
        "--loss-type",
        type=str,
        default="crossentropy",
        choices=["crossentropy", "focal"],
        help="Loss function to use for training",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    return parser.parse_args()


def collect_images_by_target_class(dataset_dir: Path) -> dict[str, list[Path]]:
    # Collect all image paths and merge source classes based on CLASS_MAPPING.
    images_by_class: dict[str, list[Path]] = {}

    for raw_class_dir in sorted(dataset_dir.iterdir()):
        if not raw_class_dir.is_dir():
            continue

        raw_name = raw_class_dir.name.lower().strip()
        if raw_name not in CLASS_MAPPING:
            continue

        target_name = CLASS_MAPPING[raw_name]
        images_by_class.setdefault(target_name, [])

        for img_path in raw_class_dir.rglob("*"):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                images_by_class[target_name].append(img_path)

    return images_by_class


def split_and_copy_dataset(images_by_class: dict[str, list[Path]], prepared_dir: Path) -> None:
    # Recreate split directory on each run to avoid stale files from previous experiments.
    if prepared_dir.exists():
        shutil.rmtree(prepared_dir)

    split_names = ["train", "val", "test"]
    for split_name in split_names:
        for class_name in sorted(images_by_class.keys()):
            (prepared_dir / split_name / class_name).mkdir(parents=True, exist_ok=True)

    for class_name, image_paths in images_by_class.items():
        if len(image_paths) < 10:
            raise ValueError(
                f"Class '{class_name}' has only {len(image_paths)} images. "
                "At least 10 images are recommended for split and training."
            )

        # 70/15/15 split: first 70/30, then split remaining 30 into 15/15.
        train_paths, temp_paths = train_test_split(
            image_paths,
            test_size=0.30,
            random_state=SEED,
            shuffle=True,
        )
        val_paths, test_paths = train_test_split(
            temp_paths,
            test_size=0.50,
            random_state=SEED,
            shuffle=True,
        )

        for src_path in train_paths:
            shutil.copy2(src_path, prepared_dir / "train" / class_name / src_path.name)
        for src_path in val_paths:
            shutil.copy2(src_path, prepared_dir / "val" / class_name / src_path.name)
        for src_path in test_paths:
            shutil.copy2(src_path, prepared_dir / "test" / class_name / src_path.name)


def oversample_train(prepared_dir: Path, class_names: list[str], target_max: int | None = None) -> dict[str, int]:
    """
    Simple oversampling: duplicate files in the training folder for underrepresented classes
    until each class reaches target_max (or the current maximum if not provided).
    Returns the resulting counts per class.
    """
    train_dir = prepared_dir / "train"
    counts = {c: len(list((train_dir / c).glob("*.*"))) for c in class_names}
    max_count = max(counts.values()) if target_max is None else int(target_max)

    for class_name, cnt in counts.items():
        if cnt >= max_count:
            continue
        src_files = list((train_dir / class_name).glob("*.*"))
        idx = 0
        while cnt < max_count and src_files:
            src = src_files[idx % len(src_files)]
            dst = train_dir / class_name / f"dup_{cnt}_{src.name}"
            shutil.copy2(src, dst)
            cnt += 1
            idx += 1

    new_counts = {c: len(list((train_dir / c).glob("*.*"))) for c in class_names}
    return new_counts



def build_datasets(prepared_dir: Path, batch_size: int):
    # Read from split folders using Keras utility to infer labels from directory names.
    train_ds = tf.keras.utils.image_dataset_from_directory(
        prepared_dir / "train",
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=batch_size,
        shuffle=True,
        seed=SEED,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        prepared_dir / "val",
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=batch_size,
        shuffle=False,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        prepared_dir / "test",
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=batch_size,
        shuffle=False,
    )

    class_names = train_ds.class_names

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    return train_ds, val_ds, test_ds, class_names


def get_class_distribution(prepared_dir: Path, class_names: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    # Count training examples per class so we can rebalance the loss during training.
    counts: dict[str, int] = {}
    for class_name in class_names:
        class_dir = prepared_dir / "train" / class_name
        counts[class_name] = len([path for path in class_dir.iterdir() if path.is_file()])

    labels = np.array([class_names.index(class_name) for class_name in class_names for _ in range(counts[class_name])])
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(class_names)),
        y=labels,
    )
    return class_weights, counts



class SparseFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma: float = 2.0, name: str = "sparse_focal_loss"):
        super().__init__(name=name)
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        ce = -y_true_one_hot * tf.math.log(y_pred)
        modulating = tf.pow(1.0 - y_pred, self.gamma)
        return tf.reduce_sum(modulating * ce, axis=-1)


def build_loss(loss_type: str, gamma: float):
    if loss_type == "focal":
        return SparseFocalLoss(gamma=gamma)

    return "sparse_categorical_crossentropy"


def build_model(num_classes: int, learning_rate: float, loss_type: str, focal_gamma: float) -> tf.keras.Model:
    # Online augmentation improves model generalization on limited datasets.
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.25),
            tf.keras.layers.RandomZoom(0.20),
            tf.keras.layers.RandomTranslation(0.12, 0.12),
            tf.keras.layers.RandomContrast(0.18),
        ],
        name="data_augmentation",
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
        name="mobilenetv2_1.00_224",
    )
    base_model.trainable = False

    # End-to-end model: augmentation -> MobileNet preprocessing -> frozen backbone -> classifier head.
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=build_loss(loss_type, focal_gamma),
        metrics=["accuracy"],
    )

    return model


def fine_tune_model(
    model: tf.keras.Model,
    learning_rate: float,
    fine_tune_layers: int,
    fine_tune_lr_mult: float,
    loss_type: str,
    focal_gamma: float,
) -> tf.keras.Model:
    # Unfreeze the upper backbone layers so the network adapts to the garbage dataset.
    try:
        base_model = model.get_layer("mobilenetv2_1.00_224")
    except ValueError:
        base_model = None

    if base_model is None:
        return model

    base_model.trainable = True

    # Unfreeze a configurable proportion of the backbone to let it adapt more.
    fine_tune_layers = min(max(1, int(fine_tune_layers)), len(base_model.layers))
    freeze_until = len(base_model.layers) - fine_tune_layers
    for index, layer in enumerate(base_model.layers):
        # keep BatchNorm layers frozen for stability
        if index < freeze_until:
            layer.trainable = False
        else:
            layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate * fine_tune_lr_mult),
        loss=build_loss(loss_type, focal_gamma),
        metrics=["accuracy"],
    )
    return model



def save_training_plots(history: tf.keras.callbacks.History, output_dir: Path) -> None:
    # Save visual training diagnostics required for reporting.
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    plt.figure(figsize=(8, 5))
    plt.plot(acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close()



def evaluate_and_save(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    class_names: list[str],
    output_dir: Path,
) -> None:
    # Convert model outputs to class IDs, then compute metrics from sklearn.
    y_true = np.concatenate([labels.numpy() for _, labels in test_ds], axis=0)
    y_prob = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()



def main() -> None:
    args = parse_args()

    dataset_dir = Path(args.dataset_dir)
    prepared_dir = Path(args.prepared_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    images_by_class = collect_images_by_target_class(dataset_dir)
    if len(images_by_class) < 5:
        raise ValueError(
            "Expected at least 5 mapped classes. "
            "Check dataset folder names and CLASS_MAPPING in train.py."
        )

    split_and_copy_dataset(images_by_class, prepared_dir)
    train_ds, val_ds, test_ds, class_names = build_datasets(prepared_dir, args.batch_size)
    class_weights, class_counts = get_class_distribution(prepared_dir, class_names)

    # Oversample underrepresented classes in the train split to match the largest class.
    new_counts = oversample_train(prepared_dir, class_names)
    print("Class distribution after oversampling:", new_counts)
    # rebuild datasets because files changed
    train_ds, val_ds, test_ds, class_names = build_datasets(prepared_dir, args.batch_size)
    class_weights, class_counts = get_class_distribution(prepared_dir, class_names)

    with (output_dir / "class_distribution.json").open("w", encoding="utf-8") as f:
        json.dump(class_counts, f, indent=2)

    model = build_model(
        num_classes=len(class_names),
        learning_rate=args.learning_rate,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
    )

    first_stage_callbacks = [
        # Stop early when validation loss stops improving and keep the best learned weights.
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=first_stage_callbacks,
        class_weight={index: float(weight) for index, weight in enumerate(class_weights)},
    )

    # Fine-tune the top part of the backbone with a lower learning rate.
    model = fine_tune_model(
        model,
        args.learning_rate,
        args.fine_tune_layers,
        args.fine_tune_lr_mult,
        args.loss_type,
        args.focal_gamma,
    )
    fine_tune_epochs = max(5, args.epochs // 3)
    second_stage_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]
    fine_tune_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=fine_tune_epochs,
        callbacks=second_stage_callbacks,
        class_weight={index: float(weight) for index, weight in enumerate(class_weights)},
    )

    model.save(output_dir / "garbage_classifier.keras")

    with (output_dir / "class_names.json").open("w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    # Merge metrics from both phases so charts show the full training story.
    combined_history = {
        key: history.history.get(key, []) + fine_tune_history.history.get(key, [])
        for key in set(history.history) | set(fine_tune_history.history)
    }
    save_training_plots(SimpleNamespace(history=combined_history), output_dir)
    evaluate_and_save(model, test_ds, class_names, output_dir)

    print("Training complete.")
    print(f"Model saved to: {output_dir / 'garbage_classifier.keras'}")
    print(f"Class labels saved to: {output_dir / 'class_names.json'}")
    print(f"Evaluation metrics saved to: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
