# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Gold training dataset
# MAGIC Construye la tabla Gold para entrenamiento/evaluación e insumos para inferencia batch.

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

silver_table = f"{full_schema}.silver_financial_sentiment_clean"
gold_table = f"{full_schema}.gold_financial_sentiment_training"
inference_table = f"{full_schema}.gold_financial_sentiment_inference_input"
dq_table = f"{full_schema}.data_quality_results"
run_log_table = f"{full_schema}.pipeline_run_log"
run_id = f"gold_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

# COMMAND ----------

silver_df = spark.table(silver_table)

split_from_source = (
    F.when(F.col("source_split_clean").isin("train", "training"), "train")
     .when(F.col("source_split_clean").isin("validation", "valid", "val"), "validation")
     .when(F.col("source_split_clean").isin("test", "testing"), "test")
)

hash_bucket = F.pmod(F.abs(F.hash(F.col("source_record_hash"))), F.lit(100))
calculated_split = (
    F.when(hash_bucket < 70, "train")
     .when(hash_bucket < 85, "validation")
     .otherwise("test")
)

gold_df = (
    silver_df
    .withColumn("split", F.coalesce(split_from_source, calculated_split))
    .withColumn("feature_text_length", F.length("text_clean"))
    .withColumn("gold_created_ts", F.current_timestamp())
    .select("source_record_hash", "text_clean", "label_normalized", "split", "feature_text_length", "source_dataset", "source_path", "gold_created_ts")
)

gold_df.write.mode("overwrite").format("delta").option("overwriteSchema", True).saveAsTable(gold_table)

# Tabla de inferencia batch: ejemplo controlado con datos test y sin etiqueta en el output operativo
inference_df = (
    gold_df
    .filter(F.col("split") == "test")
    .select("source_record_hash", "text_clean")
    .dropDuplicates(["source_record_hash"])
)

inference_df.write.mode("overwrite").format("delta").option("overwriteSchema", True).saveAsTable(inference_table)

# COMMAND ----------

summary = gold_df.groupBy("split", "label_normalized").count()
display(summary.orderBy("split", "label_normalized"))

total_gold = gold_df.count()
train_rows = gold_df.filter("split = 'train'").count()
test_rows = gold_df.filter("split = 'test'").count()
label_count = gold_df.select("label_normalized").distinct().count()

checks = [
    ("gold", "gold_has_rows", "SUCCESS" if total_gold > 0 else "FAILED", float(total_gold), "> 0", run_id, datetime.utcnow()),
    ("gold", "train_has_rows", "SUCCESS" if train_rows > 0 else "FAILED", float(train_rows), "> 0", run_id, datetime.utcnow()),
    ("gold", "test_has_rows", "SUCCESS" if test_rows > 0 else "FAILED", float(test_rows), "> 0", run_id, datetime.utcnow()),
    ("gold", "class_count", "SUCCESS" if label_count == 3 else "WARNING", float(label_count), "3 expected classes", run_id, datetime.utcnow()),
]

spark.createDataFrame(checks, "layer string, check_name string, status string, observed_value double, expected_rule string, run_id string, event_ts timestamp").write.mode("append").saveAsTable(dq_table)

if total_gold == 0 or train_rows == 0 or test_rows == 0:
    raise ValueError("Gold no tiene datos suficientes para entrenamiento/evaluación.")

spark.createDataFrame([
    (run_id, "04_gold_training_dataset", "SUCCESS", f"Gold generado con {total_gold} registros", int(total_gold), datetime.utcnow())
], "run_id string, step string, status string, message string, rows_written long, event_ts timestamp").write.mode("append").saveAsTable(run_log_table)

try:
    dbutils.jobs.taskValues.set(key="gold_rows", value=str(total_gold))
except Exception:
    pass

print(f"Gold listo. Filas: {total_gold}. Train: {train_rows}. Test: {test_rows}.")
