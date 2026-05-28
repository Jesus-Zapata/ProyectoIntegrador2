import os
import re
import unicodedata
from html import unescape

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


BASE_COLUMNS = [
    "dataset_id",
    "dataset_label",
    "source_platform",
    "split",
    "text",
    "label_normalized",
]


TEXT_COLUMN_CANDIDATES = [
    "text",
    "sentence",
    "Sentence",
    "input",
    "headline",
    "title",
    "content",
]

LABEL_COLUMN_CANDIDATES = [
    "label",
    "sentiment",
    "Sentiment",
    "output",
    "polarity",
    "target",
]


def clean_text(value):
    text = unescape(str(value))
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_column(df, candidates, column_type):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        f"No se encontró columna de {column_type}. "
        f"Columnas disponibles: {list(df.columns)}"
    )


def normalize_label(value, label_names=None):
    if pd.isna(value):
        return None

    # Si Hugging Face entrega label numérico y trae nombres de clases
    if label_names is not None:
        try:
            idx = int(value)
            if 0 <= idx < len(label_names):
                value = label_names[idx]
        except Exception:
            pass

    raw_value = str(value).strip().lower()

    positive_values = {
        "positive",
        "pos",
        "bullish",
        "mildly positive",
        "moderately positive",
        "strong positive",
    }

    negative_values = {
        "negative",
        "neg",
        "bearish",
        "mildly negative",
        "moderately negative",
        "strong negative",
    }

    neutral_values = {
        "neutral",
        "neu",
    }

    # Fallback para datasets con codificación numérica frecuente.
    # Si un dataset específico usa otro orden, luego se ajusta por fuente.
    numeric_map = {
        "0": "negative",
        "1": "neutral",
        "2": "positive",
    }

    if raw_value in positive_values:
        return "positive"

    if raw_value in negative_values:
        return "negative"

    if raw_value in neutral_values:
        return "neutral"

    if raw_value in numeric_map:
        return numeric_map[raw_value]

    return raw_value


def safe_assign_splits(df, seed=42):
    df = df.copy()

    df = df[df["text"].notna()]
    df = df[df["label_normalized"].notna()]
    df = df[df["label_normalized"].isin(["negative", "neutral", "positive"])]
    df = df[df["text"].astype(str).str.len() > 0]

    if len(df) == 0:
        return df

    class_counts = df["label_normalized"].value_counts()

    # Si hay muy pocos datos o alguna clase tiene pocos registros,
    # no usamos stratify para evitar errores.
    use_stratify = (
        df["label_normalized"].nunique() > 1
        and class_counts.min() >= 3
        and len(df) >= 20
    )

    if len(df) < 20:
        df["split"] = "train"
        return df

    train_df, temp_df = train_test_split(
        df,
        test_size=0.4,
        random_state=seed,
        stratify=df["label_normalized"] if use_stratify else None,
    )

    temp_class_counts = temp_df["label_normalized"].value_counts()
    use_temp_stratify = (
        temp_df["label_normalized"].nunique() > 1
        and temp_class_counts.min() >= 2
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=seed,
        stratify=temp_df["label_normalized"] if use_temp_stratify else None,
    )

    train_df["split"] = "train"
    val_df["split"] = "validation"
    test_df["split"] = "test"

    return pd.concat([train_df, val_df, test_df], ignore_index=True)


def get_label_names(split_dataset):
    try:
        features = split_dataset.features

        for candidate in LABEL_COLUMN_CANDIDATES:
            if candidate in features and hasattr(features[candidate], "names"):
                return features[candidate].names

    except Exception:
        pass

    return None


def load_hf_dataset_as_dataframe(repo_id):
    cache_dir = os.environ.get("HF_DATASETS_CACHE")

    dataset = load_dataset(
        repo_id,
        verification_mode="no_checks",
        cache_dir=cache_dir,
    )

    frames = []

    for split_name, split_dataset in dataset.items():
        pdf = split_dataset.to_pandas()
        pdf["_source_split"] = split_name
        pdf["_label_names"] = [get_label_names(split_dataset)] * len(pdf)
        frames.append(pdf)

    if not frames:
        raise RuntimeError(f"La fuente {repo_id} no trajo splits.")

    return pd.concat(frames, ignore_index=True)


def standardize_generic_hf_dataset(raw_df, repo_id, dataset_key):
    text_col = find_column(raw_df, TEXT_COLUMN_CANDIDATES, "texto")
    label_col = find_column(raw_df, LABEL_COLUMN_CANDIDATES, "label")

    out = pd.DataFrame()
    out["dataset_id"] = repo_id
    out["dataset_label"] = dataset_key
    out["source_platform"] = "huggingface"
    out["text"] = raw_df[text_col].map(clean_text)

    out["label_normalized"] = [
        normalize_label(value, label_names=label_names)
        for value, label_names in zip(raw_df[label_col], raw_df["_label_names"])
    ]

    out = safe_assign_splits(out)

    return out[BASE_COLUMNS]


def build_financial_sentiment_dataset(include_kaggle=False):
    jobs = [
        {
            "dataset_key": "lwrf42_financial_sentiment_dataset",
            "repo_id": "lwrf42/financial-sentiment-dataset",
        },
        {
            "dataset_key": "maguid28_combined_financial_phrasebank_twitter_news_sentiment",
            "repo_id": "maguid28/combined_financial_phrasebank_twitter_news_sentiment",
        },
    ]

    frames = []
    manifest = []

    for job in jobs:
        repo_id = job["repo_id"]
        dataset_key = job["dataset_key"]

        print("=" * 100)
        print(f"Cargando fuente: {repo_id}")

        try:
            raw_df = load_hf_dataset_as_dataframe(repo_id)

            print(f"Fuente cargada: {repo_id}")
            print(f"Filas raw: {len(raw_df)}")
            print(f"Columnas raw: {list(raw_df.columns)}")

            canonical_df = standardize_generic_hf_dataset(
                raw_df=raw_df,
                repo_id=repo_id,
                dataset_key=dataset_key,
            )

            print(f"Filas canonizadas: {len(canonical_df)}")
            print("Distribución labels:")
            print(canonical_df["label_normalized"].value_counts(dropna=False))

            if len(canonical_df) > 0:
                frames.append(canonical_df)

            manifest.append(
                {
                    "dataset_key": dataset_key,
                    "dataset_id": repo_id,
                    "source_platform": "huggingface",
                    "status": "loaded" if len(canonical_df) > 0 else "loaded_empty_after_standardization",
                    "rows": int(len(canonical_df)),
                    "error": None,
                }
            )

        except Exception as exc:
            error_message = repr(exc)

            print(f"Falló fuente: {repo_id}")
            print(f"Error: {error_message}")

            manifest.append(
                {
                    "dataset_key": dataset_key,
                    "dataset_id": repo_id,
                    "source_platform": "huggingface",
                    "status": "failed",
                    "rows": 0,
                    "error": error_message,
                }
            )

    manifest_df = pd.DataFrame(manifest)

    if not frames:
        print("Manifest de errores:")
        print(manifest_df)
        raise RuntimeError(
            "No se pudo cargar ninguna fuente de datos. "
            "Revisa el manifest impreso arriba para ver el error real por fuente."
        )

    combined_df = pd.concat(frames, ignore_index=True)

    combined_df = combined_df.drop_duplicates(
        subset=["text", "label_normalized", "split"]
    ).reset_index(drop=True)

    return combined_df, manifest_df
