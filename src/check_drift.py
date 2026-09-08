# Databricks notebook source
# src/monitoring/check_drift.py

dbutils.widgets.text("catalog", "fraud_demo_prod")
catalog = dbutils.widgets.get("catalog")

drift = spark.table(
    f"{catalog}.monitoring.fraud_detector_payload_drift_metrics")

breached = (drift
            .filter("window.start = current_date()")
            .filter("column_name != ':table'")
            .filter("js_distance > 0.2")   # threshold you tune
            .count())

dbutils.jobs.taskValues.set(key="drift_detected", value=breached > 0)
