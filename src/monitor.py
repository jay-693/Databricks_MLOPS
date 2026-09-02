# Databricks notebook source
from databricks.sdk.service.catalog import MonitorInferenceLog, MonitorInferenceLogProblemType

w.quality_monitors.create(
    table_name="fraud_demo.serving.fraud_detector_payload",
    assets_dir="/Shared/fraud_demo/monitoring",
    output_schema_name="fraud_demo.monitoring",
    inference_log=MonitorInferenceLog(
        problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
        prediction_col="prediction",
        label_col="Class",
        model_id_col="model_version",
        timestamp_col="request_ts",
        granularities=["1 day"],
    )
)
