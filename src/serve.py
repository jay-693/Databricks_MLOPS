# Databricks notebook source

import mlflow
from mlflow import MlflowClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput

# --------------------------------------------------
# MLflow / Unity Catalog
# --------------------------------------------------

mlflow.set_registry_uri("databricks-uc")

# --------------------------------------------------
# Parameters
# --------------------------------------------------

dbutils.widgets.text("catalog", "fraud_demo")
catalog = dbutils.widgets.get("catalog")

# --------------------------------------------------
# Model and endpoint
# --------------------------------------------------

registered_model_name = f"{catalog}.models.fraud_detector"
# no dev_<user>_ prefix — this only ever runs under prod
endpoint_name = "fraud-detector-endpoint"

# --------------------------------------------------
# Get champion version
# --------------------------------------------------

client = MlflowClient()
champion_mv = client.get_model_version_by_alias(
    registered_model_name, "champion")
champion_version = str(champion_mv.version)

print(f"Champion model version: {champion_version}")

# --------------------------------------------------
# Verify endpoint exists
# --------------------------------------------------

w = WorkspaceClient()
endpoint = w.serving_endpoints.get(name=endpoint_name)

print(f"Endpoint found: {endpoint.name}")
print(f"Endpoint ready state: {endpoint.state.ready}")
print(f"Endpoint config update: {endpoint.state.config_update}")

# --------------------------------------------------
# Update endpoint — single served entity, 100% traffic (default with one entity)
# --------------------------------------------------

w.serving_endpoints.update_config(
    name=endpoint_name,
    served_entities=[
        ServedEntityInput(
            entity_name=registered_model_name,
            entity_version=champion_version,
            scale_to_zero_enabled=True,
            workload_size="Small",
        ),
    ],
)

print(f"Endpoint '{endpoint_name}' update submitted.")
print(f"Now serving champion version {champion_version} (100%).")
