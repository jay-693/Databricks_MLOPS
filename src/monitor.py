from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
)

w = WorkspaceClient()

dbutils.widgets.text("catalog", "fraud_demo")
catalog = dbutils.widgets.get("catalog")

w.quality_monitors.create(
    table_name=f"{catalog}.serving.fraud_detector_payload",
    assets_dir=f"/Shared/{catalog}/monitoring",
    output_schema_name=f"{catalog}.monitoring",
    inference_log=MonitorInferenceLog(
        problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
        prediction_col="prediction",
        label_col="Class",
        model_id_col="model_version",
        timestamp_col="request_ts",
        granularities=["1 day"],
    )
)
