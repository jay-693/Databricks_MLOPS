# Databricks notebook source
from pyspark.sql import functions as F
import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")

dbutils.widgets.text("catalog", "fraud_demo_test")
catalog = dbutils.widgets.get("catalog")

registered_model_name = f"{catalog}.models.fraud_detector"
client = MlflowClient()
champion_mv = client.get_model_version_by_alias(
    registered_model_name, "champion")
model_version = str(champion_mv.version)

model_uri = f"models:/{registered_model_name}@champion"
model = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri, result_type="int")

df = spark.table(f"{catalog}.silver.txn_features")
NON_FEATURE_COLS = ["Class", "txn_id", "ingest_ts", "batch_id"]
feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

sample = df.limit(20000)

# --------------------------------------------------
# BASELINE window ("yesterday") — normal, unmodified data
# --------------------------------------------------
baseline = (sample
            .withColumn("prediction", model(*[F.col(c) for c in feature_cols]))
            .withColumn("request_ts", F.date_sub(F.current_timestamp(), 1))
            .withColumn("model_version", F.lit(model_version))
            .select(*feature_cols, "Class", "prediction", "request_ts", "model_version"))

# --------------------------------------------------
# DRIFTED window ("today") — deliberately shifted Amount + V14
# This is the synthetic drift injection: real transactions now
# look artificially larger and shifted in a key PCA component.
# --------------------------------------------------
drifted = (sample
           # amounts now much larger
           .withColumn("Amount", F.col("Amount") * 8 + 200)
           # keep engineered feature consistent
           .withColumn("amount_log", F.log1p(F.col("Amount")))
           # shift a key PCA feature hard
           .withColumn("V14", F.col("V14") + 15.0)
           .withColumn("prediction", model(*[F.col(c) for c in feature_cols]))
           .withColumn("request_ts", F.current_timestamp())
           .withColumn("model_version", F.lit(model_version))
           .select(*feature_cols, "Class", "prediction", "request_ts", "model_version"))

(baseline.write.mode("append").saveAsTable(
    f"{catalog}.serving.fraud_detector_payload"))
(drifted.write.mode("append").saveAsTable(
    f"{catalog}.serving.fraud_detector_payload"))

print(f"Logged {baseline.count()} baseline rows (yesterday) and {drifted.count()} drifted rows (today)")
