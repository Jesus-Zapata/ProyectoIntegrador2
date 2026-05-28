# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "2"
# ///
# MAGIC %md
# MAGIC # 00 - Bootstrap Unity Catalog
# MAGIC Crea el esquema, volumen y tablas de control necesarias para el pipeline DataOps/MLOps.
# MAGIC
# MAGIC Este bootstrap también prepara la estructura del volumen donde se guardan automáticamente los Parquet descargados desde Hugging Face y Kaggle.

# COMMAND ----------

from datetime import datetime

# COMMAND ----------

try:
    dbutils.widgets.text("catalog", "workspace")
    dbutils.widgets.text("schema", "financial_sentiment")
    dbutils.widgets.text("volume", "raw_landing")
    dbutils.widgets.text("project_name", "financial_sentiment_mlops")
except Exception:
    pass

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume")
project_name = dbutils.widgets.get("project_name")

full_schema = f"{catalog}.{schema}"
volume_root = f"/Volumes/{catalog}/{schema}/{volume}"
incoming_path = f"{volume_root}/incoming"
source_raw_path = f"{volume_root}/sources/raw"
source_processed_path = f"{volume_root}/sources/processed/canonical"
source_cache_path = f"{volume_root}/sources/cache"

print(f"Catalog: {catalog}")
print(f"Schema: {full_schema}")
print(f"Volume root: {volume_root}")
print(f"Parquet canónico: {source_processed_path}")

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {full_schema} COMMENT 'Proyecto Integrador 2 - DataOps y MLOps para sentimiento financiero'")

try:
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {full_schema}.{volume} COMMENT 'Volumen UC para landing, datasets fuente y parquets canónicos del pipeline'")
except Exception as e:
    print("No fue posible crear el volumen automáticamente. Puedes crearlo desde Catalog Explorer.")
    print(str(e))

for path in [
    incoming_path,
    source_raw_path,
    source_processed_path,
    f"{source_cache_path}/huggingface",
    f"{source_cache_path}/kagglehub",
    f"{volume_root}/sources/manifests",
]:
    dbutils.fs.mkdirs(path)

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {full_schema}.pipeline_run_log (
  run_id STRING,
  step STRING,
  status STRING,
  message STRING,
  rows_written BIGINT,
  event_ts TIMESTAMP
) USING DELTA
""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {full_schema}.source_dataset_manifest (
  download_batch_id STRING,
  dataset_key STRING,
  dataset_id STRING,
  source_platform STRING,
  status STRING,
  split_name STRING,
  rows_written BIGINT,
  raw_parquet_path STRING,
  canonical_parquet_path STRING,
  error_message STRING,
  event_ts TIMESTAMP
) USING DELTA
""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {full_schema}.ingestion_file_audit (
  source_path STRING,
  source_name STRING,
  file_size BIGINT,
  modification_time TIMESTAMP,
  ingestion_batch_id STRING,
  rows_loaded BIGINT,
  ingestion_ts TIMESTAMP
) USING DELTA
""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {full_schema}.data_quality_results (
  layer STRING,
  check_name STRING,
  status STRING,
  observed_value DOUBLE,
  expected_rule STRING,
  run_id STRING,
  event_ts TIMESTAMP
) USING DELTA
""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {full_schema}.model_candidate_metrics (
  training_batch_id STRING,
  model_key STRING,
  model_family STRING,
  hf_model_name STRING,
  training_mode STRING,
  run_id STRING,
  registered_model_name STRING,
  model_version STRING,
  accuracy DOUBLE,
  macro_f1 DOUBLE,
  weighted_f1 DOUBLE,
  precision_macro DOUBLE,
  recall_macro DOUBLE,
  train_rows BIGINT,
  eval_rows BIGINT,
  status STRING,
  error_message STRING,
  event_ts TIMESTAMP
) USING DELTA
""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {full_schema}.model_promotion_audit (
  promotion_batch_id STRING,
  alias STRING,
  model_key STRING,
  model_family STRING,
  hf_model_name STRING,
  registered_model_name STRING,
  model_version STRING,
  macro_f1 DOUBLE,
  accuracy DOUBLE,
  promoted_ts TIMESTAMP
) USING DELTA
""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {full_schema}.batch_predictions (
  prediction_batch_id STRING,
  source_record_hash STRING,
  text_clean STRING,
  prediction STRING,
  score DOUBLE,
  model_alias STRING,
  registered_model_name STRING,
  model_version STRING,
  prediction_ts TIMESTAMP
) USING DELTA
""")

# COMMAND ----------

run_id = f"bootstrap_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
spark.createDataFrame([
    (run_id, "00_bootstrap_uc", "SUCCESS", f"Objetos creados/verificados en {full_schema}. Volumen: {volume_root}", 0, datetime.utcnow())
], "run_id string, step string, status string, message string, rows_written long, event_ts timestamp").write.mode("append").saveAsTable(f"{full_schema}.pipeline_run_log")

try:
    dbutils.jobs.taskValues.set(key="catalog", value=catalog)
    dbutils.jobs.taskValues.set(key="schema", value=schema)
    dbutils.jobs.taskValues.set(key="volume_root", value=volume_root)
    dbutils.jobs.taskValues.set(key="source_processed_path", value=source_processed_path)
except Exception:
    pass

print("Bootstrap finalizado correctamente.")
