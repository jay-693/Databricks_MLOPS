# Databricks notebook source

import mlflow
from mlflow import MlflowClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput
from databricks.sdk.service.serving import AutoCaptureConfigInput

mlflow.set_registry_uri("databricks-uc")

dbutils.widgets.text("catalog", "fraud_demo")

catalog = dbutils.widgets.get("catalog")

registered_model_name = f"{catalog}.models.fraud_detector"

endpoint_name = "dev_vattikutivijay693_fraud-detector-endpoint"

client = MlflowClient()

champion_mv = client.get_model_version_by_alias(
    registered_model_name,
    "champion"
)

print(
    f"Updating '{endpoint_name}' "
    f"to serve champion version {champion_mv.version}"
)

w = WorkspaceClient()

# Verify endpoint exists
endpoint = w.serving_endpoints.get(
    name=endpoint_name
)

print(f"Endpoint found: {endpoint.name}")

# Update endpoint to champion model version
w.serving_endpoints.update_config(
    name=endpoint_name,
    served_entities=[
        ServedEntityInput(
            entity_name=registered_model_name,
            entity_version=str(champion_mv.version),
            scale_to_zero_enabled=True,
            workload_size="Small",
        )
    ]
)

print("Endpoint update submitted.")
