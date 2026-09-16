# Databricks notebook source
# src/monitoring/check_drift.py

dbutils.widgets.text("catalog", "fraud_demo_test")
catalog = dbutils.widgets.get("catalog")

drift = spark.table(
    f"{catalog}.monitoring.fraud_detector_payload_drift_metrics")

# If the distribution of a feature in recent production data differs sufficiently
# from the reference/baseline distribution, and its Jensen-Shannon (JS) distance is greater than 0.2,
# we consider that feature to have drifted.


breached = (drift
            .filter("window.start = current_date()")
            .filter("column_name != ':table'")
            .filter("js_distance > 0.2")   # threshold you tune
            .count())

dbutils.jobs.taskValues.set(key="drift_detected", value=breached > 0)
