# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Model promotion
# MAGIC Selecciona el mejor candidato por `macro_f1` y promueve versiones usando alias de Unity Catalog Model Registry.

# COMMAND ----------

from datetime import datetime
from pyspark.sql import functions as F
import mlflow
from mlflow.tracking import MlflowClient

try:
    dbutils.widgets.text("catalog", "workspace")
    dbutils.widgets.text("schema", "financial_sentiment")
    dbutils.widgets.text("registered_model_basename", "financial_sentiment_champion")
    dbutils.widgets.text("metric_to_optimize", "macro_f1")
except Exception:
    pass

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
registered_model_basename = dbutils.widgets.get("registered_model_basename")
metric_to_optimize = dbutils.widgets.get("metric_to_optimize")

full_schema = f"{catalog}.{schema}"
metrics_table = f"{full_schema}.model_candidate_metrics"
promotion_table = f"{full_schema}.model_promotion_audit"
run_log_table = f"{full_schema}.pipeline_run_log"
registered_model_name = f"{full_schema}.{registered_model_basename}"
promotion_batch_id = f"promotion_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# COMMAND ----------

# Toma el último training_batch_id con al menos un modelo exitoso
successful_df = spark.table(metrics_table).filter("status = 'SUCCESS' AND model_version IS NOT NULL")
latest_batch = successful_df.agg(F.max("event_ts").alias("max_ts")).collect()[0]["max_ts"]
if latest_batch is None:
    raise ValueError("No hay modelos exitosos para promover.")

latest_training_batch_id = (
    successful_df
    .orderBy(F.col("event_ts").desc())
    .select("training_batch_id")
    .first()["training_batch_id"]
)

ranking_df = (
    successful_df
    .filter(F.col("training_batch_id") == latest_training_batch_id)
    .orderBy(F.col(metric_to_optimize).desc(), F.col("accuracy").desc())
)

ranking = ranking_df.limit(3).collect()
if len(ranking) == 0:
    raise ValueError("No hay candidatos en el último lote de entrenamiento.")

alias_plan = ["Champion", "Challenger_1", "Challenger_2"]

rows = []
for idx, row in enumerate(ranking):
    alias = alias_plan[idx]
    version = int(row["model_version"])
    client.set_registered_model_alias(registered_model_name, alias, version)
    client.set_model_version_tag(registered_model_name, str(version), "promotion_alias", alias)
    client.set_model_version_tag(registered_model_name, str(version), "promotion_batch_id", promotion_batch_id)
    client.set_model_version_tag(registered_model_name, str(version), "optimized_metric", metric_to_optimize)

    rows.append((
        promotion_batch_id,
        alias,
        row["model_key"],
        row["model_family"],
        row["hf_model_name"],
        registered_model_name,
        str(version),
        float(row["macro_f1"]) if row["macro_f1"] is not None else None,
        float(row["accuracy"]) if row["accuracy"] is not None else None,
        datetime.utcnow(),
    ))

schema_audit = "promotion_batch_id string, alias string, model_key string, model_family string, hf_model_name string, registered_model_name string, model_version string, macro_f1 double, accuracy double, promoted_ts timestamp"
spark.createDataFrame(rows, schema_audit).write.mode("append").saveAsTable(promotion_table)

spark.createDataFrame([
    (promotion_batch_id, "06_promote_model", "SUCCESS", f"Promoción realizada sobre {registered_model_name}", int(len(rows)), datetime.utcnow())
], "run_id string, step string, status string, message string, rows_written long, event_ts timestamp").write.mode("append").saveAsTable(run_log_table)

try:
    dbutils.jobs.taskValues.set(key="promotion_batch_id", value=promotion_batch_id)
    dbutils.jobs.taskValues.set(key="champion_version", value=rows[0][6])
    dbutils.jobs.taskValues.set(key="champion_model_key", value=rows[0][2])
except Exception:
    pass

display(spark.createDataFrame(rows, schema_audit))
print(f"Champion: {rows[0][2]} version {rows[0][6]}")
