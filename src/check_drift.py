# Databricks notebook source
# src/monitoring/check_drift.py

from pyspark.sql import functions as F
dbutils.widgets.text("catalog", "fraud_demo_test")
catalog = dbutils.widgets.get("catalog")


drift = spark.table(
    f"{catalog}.monitoring.fraud_detector_payload_drift_metrics"
)

latest_window = (
    drift
    .select(F.max(F.col("window.start")).alias("latest_window"))
    .collect()[0]["latest_window"]
)

breached = (
    drift
    .filter(F.col("window.start") == latest_window)
    .filter(F.col("column_name") != ":table")
    .filter(
        (F.col("ks_test.statistic") > 0.2) &
        (F.col("ks_test.pvalue") < 0.05)
    )
    .count()
)


dbutils.jobs.taskValues.set(key="drift_detected", value=breached > 0)
