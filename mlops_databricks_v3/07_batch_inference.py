# Databricks notebook source
# MAGIC %md
# MAGIC # 07 - Batch inference
# MAGIC Consume el alias `Champion` del Model Registry y genera predicciones batch sobre una tabla Gold.

# COMMAND ----------

from datetime import datetime
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F

try:
    dbutils.widgets.text("catalog", "workspace")
    dbutils.widgets.text("schema", "financial_sentiment")
    dbutils.widgets.text("registered_model_basename", "financial_sentiment_champion")
    dbutils.widgets.text("model_alias", "Champion")
except Exception:
    pass

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
registered_model_basename = dbutils.widgets.get("registered_model_basename")
model_alias = dbutils.widgets.get("model_alias")

full_schema = f"{catalog}.{schema}"
inference_table = f"{full_schema}.gold_financial_sentiment_inference_input"
predictions_table = f"{full_schema}.batch_predictions"
run_log_table = f"{full_schema}.pipeline_run_log"
registered_model_name = f"{full_schema}.{registered_model_basename}"
prediction_batch_id = f"predict_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# COMMAND ----------

# DBTITLE 1,Cell 3
# Alternative 1: Use AI_QUERY() for batch inference via model serving endpoint
# This requires a model serving endpoint to be deployed first
# See: https://docs.databricks.com/en/machine-learning/model-serving/index.html

# Alternative 2: If model artifacts are in DBFS and accessible, load from there directly
# Get the actual artifact location from the model version
model_version_info = client.get_model_version_by_alias(registered_model_name, model_alias)
model_version = model_version_info.version

# For Unity Catalog models, use the model serving SQL function approach
# This query would work if a serving endpoint named 'financial_sentiment_endpoint' exists:
# predictions_df = spark.sql(f"""
#   SELECT 
#     source_record_hash,
#     text_clean,
#     ai_query('financial_sentiment_endpoint', text_clean) as prediction_result,
#     '{prediction_batch_id}' as prediction_batch_id,
#     '{model_alias}' as model_alias,
#     '{registered_model_name}' as registered_model_name,
#     '{model_version}' as model_version,
#     current_timestamp() as prediction_ts
#   FROM {inference_table}
# """)

# For now, read the input data and prepare for manual prediction
input_df = spark.table(inference_table).select("source_record_hash", "text_clean").dropDuplicates(["source_record_hash"])

if input_df.count() == 0:
    raise ValueError("No hay registros para inferencia batch.")

print(f"Input records ready: {input_df.count()}")
print(f"\nTo complete batch inference, you need to:")
print(f"1. Deploy the model '{registered_model_name}@{model_alias}' to a serving endpoint")
print(f"2. Use AI_QUERY() SQL function to invoke the endpoint for batch predictions")
print(f"3. Or run this notebook on a cluster with direct DBFS access (not Serverless)")

display(input_df.limit(5))
