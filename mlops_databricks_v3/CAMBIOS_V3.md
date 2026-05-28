# Cambios principales frente a v2

1. Se agregó `01_download_source_parquets.py`.
2. La descarga de Hugging Face/Kaggle es automática dentro del Job.
3. Los datasets se guardan como Parquet dentro del volumen de Unity Catalog.
4. Bronze ahora lee desde `/sources/processed/canonical`, no desde CSV manual en `/incoming`.
5. Se agregó la tabla `source_dataset_manifest` para auditar descargas, rutas, filas y errores.
6. Se actualizó el orden del Job hasta `08_pipeline_evidence`.
