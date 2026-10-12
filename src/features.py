# Databricks notebook source
# MAGIC %pip install databricks-feature-engineering
# MAGIC %restart_python
from pyspark.sql import functions as F
from databricks.feature_engineering import FeatureEngineeringClient


fe = FeatureEngineeringClient()

dbutils.widgets.text("catalog", "fraud_demo")

catalog = dbutils.widgets.get("catalog")

sources = {
    "fraud_features":      f"{catalog}.bronze.creditcardtransactions",
    "fraud_features_test": f"{catalog}.bronze.creditcardtransactions_test",
}

for target, source in sources.items():
    df = spark.table(source).withColumn(
        "txn_id", F.monotonically_increasing_id())
    fe.create_table(
        name=f"{catalog}.silver.{target}",
        primary_keys=["txn_id"],
        df=df,
        description=f"Engineered features for fraud model ({target})",
    )
