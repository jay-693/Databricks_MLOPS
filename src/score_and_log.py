# Databricks notebook source
import mlflow
from mlflow import MlflowClient
from pyspark.sql import functions as F

mlflow.set_registry_uri("databricks-uc")

dbutils.widgets.text("catalog", "fraud_demo_prod")
dbutils.widgets.text("batch_id", "manual")
catalog = dbutils.widgets.get("catalog")
batch_id = dbutils.widgets.get("batch_id")

registered_model_name = f"{catalog}.models.fraud_detector"

client = MlflowClient()
champion_mv = client.get_model_version_by_alias(
    registered_model_name, "champion")
model_version = str(champion_mv.version)

model_uri = f"models:/{registered_model_name}@champion"
model = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri, result_type="int")

# Score a slice of the feature table — in production this would be genuinely new traffic;
# here we simulate it by pulling a batch that hasn't been logged yet
df = spark.table(f"{catalog}.silver.txn_features")

NON_FEATURE_COLS = ["Class", "txn_id", "ingest_ts", "batch_id"]
feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

scored = (df
          .withColumn("prediction", model(*[F.col(c) for c in feature_cols]))
          .withColumn("request_ts", F.current_timestamp())
          .withColumn("model_version", F.lit(model_version))
          .select(*feature_cols, "Class", "prediction", "request_ts", "model_version")
          )

(scored.write
    .mode("append")
    .saveAsTable(f"{catalog}.serving.fraud_detector_payload"))

print(f"Logged {scored.count()} scored rows against champion v{model_version}")
