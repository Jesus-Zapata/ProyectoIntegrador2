from __future__ import annotations

import math
import os
from typing import Any

import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from tqdm.auto import tqdm

from .text_utils import LABEL_TO_ID, evaluate_predictions


def _build_model_label_maps(model_label_order: list[str]) -> tuple[dict[int, int], dict[int, int]]:
    model_to_canonical_id = {
        model_label_id: LABEL_TO_ID[label_name]
        for model_label_id, label_name in enumerate(model_label_order)
    }
    canonical_to_model_id = {canonical_id: model_id for model_id, canonical_id in model_to_canonical_id.items()}
    return canonical_to_model_id, model_to_canonical_id


def _map_label_ids(label_ids: list[int], mapping: dict[int, int]) -> list[int]:
    return [mapping[int(label_id)] for label_id in label_ids]


def build_transformer_trainer(
    df: pd.DataFrame,
    model_name: str,
    model_label_order: list[str],
    output_dir: str,
    max_length: int = 128,
    epochs: int = 2,
    train_batch_size: int = 16,
    eval_batch_size: int = 32,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    early_stopping_patience: int = 2,
    seed: int = 42,
) -> tuple[Any, Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    canonical_to_model_id, model_to_canonical_id = _build_model_label_maps(model_label_order)
    split_datasets = {}

    for split_name in ["train", "validation", "test"]:
        split_df = df[df["split"] == split_name][["text", "label_id"]].copy()
        dataset = HFDataset.from_pandas(split_df, preserve_index=False)
        dataset = dataset.map(
            lambda batch: {
                **tokenizer(batch["text"], truncation=True, max_length=max_length),
                "labels": _map_label_ids(batch["label_id"], canonical_to_model_id),
            },
            batched=True,
            remove_columns=dataset.column_names,
        )
        split_datasets[split_name] = dataset

    datasets = DatasetDict(split_datasets)
    bf16_enabled = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    steps_per_epoch = math.ceil(len(datasets["train"]) / max(train_batch_size, 1))
    warmup_steps = int(steps_per_epoch * epochs * warmup_ratio)

    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        logging_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        report_to="none",
        dataloader_num_workers=min(os.cpu_count() or 1, 4),
        seed=seed,
        data_seed=seed,
        bf16=bf16_enabled,
        fp16=torch.cuda.is_available() and not bf16_enabled,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=lambda prediction: {
            key: float(value)
            for key, value in evaluate_predictions(
                _map_label_ids(prediction.label_ids.tolist(), model_to_canonical_id),
                _map_label_ids(prediction.predictions.argmax(axis=-1).tolist(), model_to_canonical_id),
            ).items()
            if isinstance(value, float)
        },
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )
    return trainer, datasets, tokenizer


def evaluate_transformer_checkpoint(
    model_name: str,
    model_label_order: list[str],
    df: pd.DataFrame,
    split: str = "test",
    max_length: int = 128,
    batch_size: int = 32,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    _, model_to_canonical_id = _build_model_label_maps(model_label_order)
    split_df = df[df["split"] == split][["text", "label_id"]].reset_index(drop=True)

    all_true: list[int] = []
    all_pred: list[int] = []
    all_texts: list[str] = []

    model.eval()
    with torch.inference_mode():
        for start in tqdm(range(0, len(split_df), batch_size), desc=f"evaluating {model_name}", leave=False):
            batch_df = split_df.iloc[start : start + batch_size]
            texts = batch_df["text"].tolist()
            labels = batch_df["label_id"].astype(int).tolist()
            batch = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            predicted_ids = logits.argmax(dim=-1).cpu().tolist()

            all_true.extend(labels)
            all_pred.extend(_map_label_ids(predicted_ids, model_to_canonical_id))
            all_texts.extend(texts)

    results = evaluate_predictions(all_true, all_pred, texts=all_texts)
    return results
