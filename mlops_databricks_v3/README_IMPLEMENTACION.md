# Implementación DataOps/MLOps en Databricks Free Edition - versión con descarga automática a Parquet

## Objetivo

Implementar un flujo automatizado en Databricks Free Edition para:

1. Descargar automáticamente datasets desde Hugging Face y Kaggle.
2. Guardar en el volumen de Unity Catalog los snapshots raw y los Parquet canónicos ya estandarizados.
3. Cargar Bronze desde Parquet, sin subir CSV manualmente.
4. Construir Silver y Gold en tablas Delta administradas por Unity Catalog.
5. Entrenar/evaluar los tres mejores modelos del repositorio con MLflow.
6. Registrar versiones en Unity Catalog Model Registry.
7. Promover el mejor modelo con alias `Champion` y mantener `Challenger_1` y `Challenger_2`.
8. Ejecutar inferencia batch consumiendo el alias `Champion`.

## Fuentes automatizadas

El notebook `01_download_source_parquets.py` descarga estas fuentes:

- Hugging Face: `lwrf42/financial-sentiment-dataset`.
- Hugging Face: `Kenpache/multilingual-financial-sentiment`, filtrado a inglés.
- Hugging Face: `maguid28/combined_financial_phrasebank_twitter_news_sentiment`.
- Kaggle: `sbhatti/financial-sentiment-analysis`.

## Estructura en el volumen

Ruta raíz sugerida:

```text
/Volumes/workspace/financial_sentiment/raw_landing
```

El pipeline crea automáticamente estas carpetas:

```text
/Volumes/workspace/financial_sentiment/raw_landing/sources/raw/huggingface/<dataset>/<split>
/Volumes/workspace/financial_sentiment/raw_landing/sources/raw/kaggle/<dataset>/<split>
/Volumes/workspace/financial_sentiment/raw_landing/sources/processed/canonical/<dataset>
/Volumes/workspace/financial_sentiment/raw_landing/sources/processed/combined/financial_sentiment_all
/Volumes/workspace/financial_sentiment/raw_landing/sources/cache/huggingface
/Volumes/workspace/financial_sentiment/raw_landing/sources/cache/kagglehub
/Volumes/workspace/financial_sentiment/raw_landing/sources/manifests
```

La carpeta importante para Bronze es:

```text
/Volumes/workspace/financial_sentiment/raw_landing/sources/processed/canonical
```

Ahí queda un Parquet canónico por dataset, con columnas estandarizadas:

```text
dataset_id,dataset_label,source_platform,split,text,label_normalized,download_batch_id,source_record_hash
```

## Tabla de auditoría de descarga

El notebook de descarga deja evidencia en:

```sql
SELECT *
FROM workspace.financial_sentiment.source_dataset_manifest
ORDER BY event_ts DESC;
```

Esta tabla muestra por fuente:

- `dataset_key`
- `dataset_id`
- `source_platform`
- `status`
- `split_name`
- `rows_written`
- `raw_parquet_path`
- `canonical_parquet_path`
- `error_message`
- `event_ts`

## Flujo de capas

### 1. Source landing en volumen

La fuente real ya no es un CSV manual. El Job descarga datasets públicos y los guarda en Parquet dentro del volumen de Unity Catalog.

### 2. Bronze

Tabla:

```text
workspace.financial_sentiment.bronze_financial_sentiment_raw
```

Carga desde Parquet canónico y conserva trazabilidad:

- `source_record_hash`
- `source_dataset`
- `source_dataset_id`
- `source_platform`
- `source_download_batch_id`
- `source_path`
- `ingestion_batch_id`
- `ingestion_ts`

La ingesta es idempotente porque evita cargar registros ya existentes usando `source_record_hash`.

### 3. Silver

Tabla:

```text
workspace.financial_sentiment.silver_financial_sentiment_clean
```

Limpia texto, normaliza etiquetas, elimina duplicados y separa registros inválidos en:

```text
workspace.financial_sentiment.silver_financial_sentiment_quarantine
```

### 4. Gold

Tabla:

```text
workspace.financial_sentiment.gold_financial_sentiment_training
```

Contiene el dataset listo para entrenamiento/evaluación con splits reproducibles.

Tabla de inferencia:

```text
workspace.financial_sentiment.gold_financial_sentiment_inference_input
```

## Orden del Job

Crea un Job secuencial con estas tareas tipo Notebook:

```text
00_bootstrap_uc
01_download_source_parquets
02_bronze_ingestion
03_silver_transform
04_gold_training_dataset
05_train_top3_models_mlflow
06_promote_model
07_batch_inference
08_pipeline_evidence
```

## Parámetros recomendados del Job

```text
catalog = workspace
schema = financial_sentiment
volume = raw_landing
enable_huggingface = true
enable_kaggle = true
force_refresh = false
fail_if_no_sources = true
fail_if_any_source_fails = false
source_mode = canonical_parquet
experiment_name = /Shared/financial_sentiment_mlops
registered_model_basename = financial_sentiment_champion
metric_to_optimize = macro_f1
```

## Kaggle

Kaggle puede pedir credenciales aunque el dataset sea público. En ese caso, usa los parámetros del Job:

```text
kaggle_username = <tu_usuario_kaggle>
kaggle_key = <tu_api_key_kaggle>
```

No subas esas credenciales al repositorio. Si tu workspace permite secretos, mejor referencia esos valores desde secrets. Si no, ponlos solo como parámetro temporal del Job y bórralos después.

## Sustentación corta

Se implementó DataOps porque el pipeline automatiza la adquisición de datos desde fuentes externas, guarda snapshots raw y datasets canónicos en Parquet dentro de un volumen de Unity Catalog, registra un manifiesto de auditoría, carga Bronze de forma idempotente y construye Silver/Gold con reglas de calidad y trazabilidad.

Se implementó MLOps porque el Job entrena/evalúa los tres mejores modelos del repositorio, registra parámetros, métricas, artefactos y versiones en MLflow, promueve automáticamente el mejor candidato mediante alias del Model Registry de Unity Catalog y ejecuta inferencia batch consumiendo el alias `Champion`.

## Evidencias para mostrar

```sql
SHOW TABLES IN workspace.financial_sentiment;

SELECT * FROM workspace.financial_sentiment.source_dataset_manifest ORDER BY event_ts DESC;
SELECT * FROM workspace.financial_sentiment.pipeline_run_log ORDER BY event_ts DESC;
SELECT * FROM workspace.financial_sentiment.ingestion_file_audit ORDER BY ingestion_ts DESC;
SELECT * FROM workspace.financial_sentiment.data_quality_results ORDER BY event_ts DESC;
SELECT * FROM workspace.financial_sentiment.model_candidate_metrics ORDER BY event_ts DESC;
SELECT * FROM workspace.financial_sentiment.model_promotion_audit ORDER BY promoted_ts DESC;
SELECT * FROM workspace.financial_sentiment.batch_predictions ORDER BY prediction_ts DESC;
```
