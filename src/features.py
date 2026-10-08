# Databricks notebook source
# MAGIC %pip install databricks-feature-engineering
# MAGIC %restart_python
from pyspark.sql import functions as F
from databricks.feature_engineering import FeatureEngineeringClient


fe = FeatureEngineeringClient()

dbutils.widgets.text("catalog", "fraud_demo")

catalog = dbutils.widgets.get("catalog")

bronze = spark.table(f"{catalog}.bronze.creditcardtransactions")

features = (bronze
            .select("*")
            .withColumn("txn_id", F.monotonically_increasing_id())
            )

fe.create_table(
    name=f"{catalog}.silver.fraud_features",
    primary_keys=["txn_id"],
    df=features,
    description="Engineered features for fraud model"
)
