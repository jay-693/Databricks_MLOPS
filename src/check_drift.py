# Databricks notebook source
# src/monitoring/check_drift.py
drift = spark.table(
    "fraud_demo.monitoring.fraud_detector_payload_drift_metrics")

breached = (drift
            .filter("window.start = current_date() - 1")
            .filter("column_name != ':table'")
            .filter("js_distance > 0.2")   # threshold you tune
            .count())

dbutils.jobs.taskValues.set(key="drift_detected", value=breached > 0)
