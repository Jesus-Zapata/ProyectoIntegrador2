# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Silver transformation
# MAGIC Limpia texto, normaliza etiquetas, elimina duplicados y registra validaciones de calidad.

# COMMAND ----------

from datetime import datetime
from pyspark.sql import functions as F

try:
    dbutils.widgets.text("catalog", "workspace")
    dbutils.widgets.text("schema", "financial_sentiment")
except Exception:
    pass

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
full_schema = f"{catalog}.{schema}"

bronze_table = f"{full_schema}.bronze_financial_sentiment_raw"
silver_table = f"{full_schema}.silver_financial_sentiment_clean"
quarantine_table = f"{full_schema}.silver_financial_sentiment_quarantine"
dq_table = f"{full_schema}.data_quality_results"
run_log_table = f"{full_schema}.pipeline_run_log"
run_id = f"silver_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

# COMMAND ----------

bronze_df = spark.table(bronze_table)

clean_text = F.trim(F.regexp_replace(F.lower(F.col("raw_text")), r"\s+", " "))
clean_text = F.regexp_replace(clean_text, r"https?://\S+|www\.\S+", "")
clean_text = F.regexp_replace(clean_text, r"[^a-zA-Z0-9áéíóúñüÁÉÍÓÚÑÜ$%.,;:!?\- ]", "")
clean_text = F.trim(F.regexp_replace(clean_text, r"\s+", " "))

raw_label_norm = F.lower(F.trim(F.col("raw_label")))
label_normalized = (
    F.when(raw_label_norm.isin("positive", "pos", "1", "bullish"), "positive")
     .when(raw_label_norm.isin("negative", "neg", "-1", "bearish"), "negative")
     .when(raw_label_norm.isin("neutral", "neu", "0"), "neutral")
     .otherwise(None)
)

base = (
    bronze_df
    .withColumn("text_clean", clean_text)
    .withColumn("label_normalized", label_normalized)
    .withColumn("source_split_clean", F.lower(F.trim(F.col("source_split"))))
    .withColumn("source_record_hash", F.coalesce(F.col("source_record_hash"), F.sha2(F.concat_ws("||", F.col("text_clean"), F.coalesce(F.col("label_normalized"), F.lit("unknown"))), 256)))
)

valid_df = (
    base
    .filter(F.col("text_clean").isNotNull())
    .filter(F.length("text_clean") >= 5)
    .filter(F.col("label_normalized").isin("negative", "neutral", "positive"))
    .dropDuplicates(["source_record_hash"])
    .select(
        "source_record_hash", "text_clean", "label_normalized", "source_split_clean",
        "source_dataset", "source_path", "ingestion_batch_id", "ingestion_ts"
    )
)

quarantine_df = (
    base
    .filter((F.col("text_clean").isNull()) | (F.length("text_clean") < 5) | (~F.col("label_normalized").isin("negative", "neutral", "positive")) | F.col("label_normalized").isNull())
    .withColumn("quarantine_reason", F.lit("Texto vacío/corto o etiqueta inválida"))
)

valid_df.write.mode("overwrite").format("delta").option("overwriteSchema", True).saveAsTable(silver_table)
quarantine_df.write.mode("overwrite").format("delta").option("overwriteSchema", True).saveAsTable(quarantine_table)

# COMMAND ----------

total_bronze = bronze_df.count()
total_silver = valid_df.count()
total_quarantine = quarantine_df.count()
distinct_labels = valid_df.select("label_normalized").distinct().count()

checks = [
    ("silver", "bronze_has_rows", "SUCCESS" if total_bronze > 0 else "FAILED", float(total_bronze), "> 0", run_id, datetime.utcnow()),
    ("silver", "silver_has_rows", "SUCCESS" if total_silver > 0 else "FAILED", float(total_silver), "> 0", run_id, datetime.utcnow()),
    ("silver", "valid_label_count", "SUCCESS" if distinct_labels == 3 else "WARNING", float(distinct_labels), "3 expected classes", run_id, datetime.utcnow()),
    ("silver", "quarantine_rows", "SUCCESS", float(total_quarantine), "informativo", run_id, datetime.utcnow()),
]

spark.createDataFrame(checks, "layer string, check_name string, status string, observed_value double, expected_rule string, run_id string, event_ts timestamp").write.mode("append").saveAsTable(dq_table)

if total_silver == 0:
    raise ValueError("La tabla Silver quedó vacía. Revisa la ingesta Bronze y las etiquetas.")

spark.createDataFrame([
    (run_id, "03_silver_transform", "SUCCESS", f"Silver generado con {total_silver} registros válidos y {total_quarantine} en cuarentena", int(total_silver), datetime.utcnow())
], "run_id string, step string, status string, message string, rows_written long, event_ts timestamp").write.mode("append").saveAsTable(run_log_table)

try:
    dbutils.jobs.taskValues.set(key="silver_rows", value=str(total_silver))
except Exception:
    pass

print(f"Silver listo. Registros válidos: {total_silver}. Cuarentena: {total_quarantine}.")
