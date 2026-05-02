from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences


LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "requirements.txt").exists():
            return candidate
    return start


ROOT = find_project_root()
DATA_PATH = ROOT / "data" / "processed" / "financial_sentiment_combined.csv"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    csv_path = Path(csv_path or DATA_PATH)
    df = pd.read_csv(csv_path)
    if "label_id" not in df.columns:
        df["label_id"] = df["label_normalized"].map(LABEL_TO_ID)
    df["label_id"] = df["label_id"].astype(np.int64)
    return df.reset_index(drop=True)


def summarize_splits(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("split")
        .agg(
            rows=("text", "size"),
            mean_words=("text", lambda values: values.str.split().str.len().mean()),
            negative=("label_normalized", lambda values: (values == "negative").sum()),
            neutral=("label_normalized", lambda values: (values == "neutral").sum()),
            positive=("label_normalized", lambda values: (values == "positive").sum()),
        )
        .round(2)
    )


def evaluate_predictions(y_true: list[int], y_pred: list[int], texts: list[str] | None = None) -> dict[str, Any]:
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average="weighted",
        zero_division=0,
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=[ID_TO_LABEL[index] for index in [0, 1, 2]],
        zero_division=0,
        output_dict=True,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "report": pd.DataFrame(report).transpose(),
        "confusion_matrix": pd.DataFrame(
            matrix,
            index=[ID_TO_LABEL[index] for index in [0, 1, 2]],
            columns=[ID_TO_LABEL[index] for index in [0, 1, 2]],
        ),
    }

    if texts is not None:
        results["predictions"] = pd.DataFrame(
            {
                "text": texts,
                "label": [ID_TO_LABEL[index] for index in y_true],
                "prediction": [ID_TO_LABEL[index] for index in y_pred],
            }
        )

    return results


def metrics_table(metrics: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {"metric": key, "value": value}
        for key, value in metrics.items()
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)
    ]
    return pd.DataFrame(rows)


def plot_history(history: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["epoch"], history["train_loss"], marker="o", label="train")
    axes[0].plot(history["epoch"], history["validation_loss"], marker="o", label="validation")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["validation_accuracy"], marker="o", label="accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].legend()

    plt.tight_layout()


def download_hf_file(repo_id: str, filename: str) -> str:
    return hf_hub_download(repo_id, filename)


def load_pickle_tokenizer(repo_id: str, filename: str = "my_tokenizer.pkl") -> Any:
    tokenizer_path = download_hf_file(repo_id, filename)
    with open(tokenizer_path, "rb") as handle:
        return pickle.load(handle)


def keras_text_splits(
    df: pd.DataFrame,
    tokenizer: Any,
    max_length: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for split_name in ["train", "validation", "test"]:
        split_df = df[df["split"] == split_name].copy()
        texts = split_df["text"].tolist()
        sequences = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")
        labels = split_df["label_id"].to_numpy(dtype=np.int64)
        result[split_name] = (padded, labels)

    return result


def replace_keras_classifier(
    base_model: Any,
    num_classes: int = 3,
    activation: str = "softmax",
    name: str = "transfer_head",
) -> Any:
    inputs = keras.Input(
        shape=tuple(base_model.inputs[0].shape[1:]),
        dtype=base_model.inputs[0].dtype,
        name="transfer_input",
    )
    features = inputs
    for layer in base_model.layers[:-1]:
        features = layer(features)

    outputs = keras.layers.Dense(num_classes, activation=activation, name=name)(features)
    return keras.Model(inputs=inputs, outputs=outputs)


def set_trainable_layers(model: Any, trainable_count_from_end: int) -> None:
    total_layers = len(model.layers)
    threshold = max(total_layers - trainable_count_from_end, 0)
    for index, layer in enumerate(model.layers):
        layer.trainable = index >= threshold


def balanced_class_weights(y_train: np.ndarray) -> dict[int, float]:
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(label): float(weight) for label, weight in zip(classes, weights)}


def build_training_callbacks(
    early_stopping_patience: int,
    reduce_lr_patience: int,
) -> list[keras.callbacks.Callback]:
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            min_delta=1e-3,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def train_keras_transfer_model(
    model: Any,
    train_data: tuple[np.ndarray, np.ndarray],
    validation_data: tuple[np.ndarray, np.ndarray],
    head_epochs: int = 2,
    full_epochs: int = 2,
    head_learning_rate: float = 1e-3,
    full_learning_rate: float = 1e-4,
    batch_size: int = 64,
    unfreeze_layers_from_end: int = 4,
    use_class_weight: bool = True,
    early_stopping_patience: int = 2,
    reduce_lr_patience: int = 1,
) -> pd.DataFrame:
    x_train, y_train = train_data
    x_valid, y_valid = validation_data
    class_weight = balanced_class_weights(y_train) if use_class_weight else None

    set_trainable_layers(model, trainable_count_from_end=1)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=head_learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    head_history = model.fit(
        x_train,
        y_train,
        validation_data=(x_valid, y_valid),
        epochs=head_epochs,
        batch_size=batch_size,
        callbacks=build_training_callbacks(
            early_stopping_patience=early_stopping_patience,
            reduce_lr_patience=reduce_lr_patience,
        ),
        class_weight=class_weight,
        verbose=1,
    )

    set_trainable_layers(model, trainable_count_from_end=unfreeze_layers_from_end)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=full_learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    full_history = model.fit(
        x_train,
        y_train,
        validation_data=(x_valid, y_valid),
        epochs=full_epochs,
        batch_size=batch_size,
        callbacks=build_training_callbacks(
            early_stopping_patience=early_stopping_patience,
            reduce_lr_patience=reduce_lr_patience,
        ),
        class_weight=class_weight,
        verbose=1,
    )

    history_rows: list[dict[str, float]] = []
    epoch_index = 1
    for history in [head_history.history, full_history.history]:
        epoch_count = len(history["loss"])
        for offset in range(epoch_count):
            history_rows.append(
                {
                    "epoch": epoch_index,
                    "train_loss": float(history["loss"][offset]),
                    "validation_loss": float(history["val_loss"][offset]),
                    "validation_accuracy": float(history["val_accuracy"][offset]),
                }
            )
            epoch_index += 1

    return pd.DataFrame(history_rows)


def evaluate_keras_model(
    model: Any,
    x_data: np.ndarray,
    y_true: np.ndarray,
    texts: list[str] | None = None,
) -> dict[str, Any]:
    probabilities = model.predict(x_data, verbose=0)
    y_pred = probabilities.argmax(axis=1).tolist()
    results = evaluate_predictions(y_true.tolist(), y_pred, texts=texts)
    return results
