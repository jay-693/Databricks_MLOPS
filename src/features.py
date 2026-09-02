# Databricks notebook source
# MAGIC %pip install databricks-feature-engineering
# MAGIC %restart_python
from pyspark.sql import functions as F
from databricks.feature_engineering import FeatureEngineeringClient


fe = FeatureEngineeringClient()

dbutils.widgets.text("catalog", "fraud_demo")

catalog = dbutils.widgets.get("catalog")

bronze = spark.table(f"{catalog}.bronze.transactions")

features = (bronze
            .select("Time", "Amount", *[f"V{i}" for i in range(1, 29)], "Class")
            .withColumn("amount_log", F.log1p("Amount"))
            .withColumn("hour_of_day", (F.col("Time") % 86400) / 3600)
            .withColumn("txn_id", F.monotonically_increasing_id())
            )

fe.create_table(
    name=f"{catalog}.silver.txn_features",
    primary_keys=["txn_id"],
    df=features,
    description="Engineered features for fraud model"
)
