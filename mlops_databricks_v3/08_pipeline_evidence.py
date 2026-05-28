# Databricks notebook source
# MAGIC %md
# MAGIC # 08 - Pipeline evidence
# MAGIC Consulta evidencia final del proceso DataOps/MLOps: logs, calidad, métricas, promoción e inferencia.

# COMMAND ----------

try:
    dbutils.widgets.text("catalog", "workspace")
    dbutils.widgets.text("schema", "financial_sentiment")
except Exception:
    pass

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
full_schema = f"{catalog}.{schema}"

# COMMAND ----------

print("Tablas creadas en Unity Catalog")
display(spark.sql(f"SHOW TABLES IN {full_schema}"))

# COMMAND ----------

print("Manifiesto de descarga automática de Hugging Face/Kaggle")
display(spark.table(f"{full_schema}.source_dataset_manifest").orderBy("event_ts", ascending=False))

# COMMAND ----------

print("Ejecución del pipeline")
display(spark.table(f"{full_schema}.pipeline_run_log").orderBy("event_ts", ascending=False))

# COMMAND ----------

print("Resultados de calidad por capa")
display(spark.table(f"{full_schema}.data_quality_results").orderBy("event_ts", ascending=False))

# COMMAND ----------

print("Métricas de candidatos MLflow")
display(spark.table(f"{full_schema}.model_candidate_metrics").orderBy("event_ts", ascending=False))

# COMMAND ----------

print("Promoción de modelos")
display(spark.table(f"{full_schema}.model_promotion_audit").orderBy("promoted_ts", ascending=False))

# COMMAND ----------

print("Predicciones batch")
display(spark.table(f"{full_schema}.batch_predictions").orderBy("prediction_ts", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sustentación corta
# MAGIC
# MAGIC Se implementó un pipeline DataOps/MLOps en Databricks Free Edition usando Lakeflow Jobs, serverless compute, Unity Catalog, Delta tables y MLflow.
# MAGIC
# MAGIC **DataOps:** la carga parte de una tarea automática que descarga datasets desde Hugging Face y Kaggle, guarda snapshots raw y Parquet canónico en un volumen de Unity Catalog, registra un manifiesto de auditoría y carga Bronze desde esos Parquet. Luego transforma y valida datos en Silver, dejando registros inválidos en cuarentena. Finalmente construye Gold con splits reproducibles para entrenamiento, validación, prueba e inferencia batch.
# MAGIC
# MAGIC **MLOps:** el job toma la tabla Gold como fuente única de entrenamiento/evaluación, ejecuta los tres mejores modelos del repositorio, registra parámetros, métricas, artefactos y versiones en MLflow, y promueve automáticamente el mejor por `macro_f1` usando alias del Model Registry de Unity Catalog. La inferencia batch consume el alias `Champion`, por lo que el proceso queda desacoplado de una versión fija del modelo.
