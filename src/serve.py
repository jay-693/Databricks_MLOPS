# Databricks notebook source

import mlflow
from mlflow import MlflowClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput

mlflow.set_registry_uri("databricks-uc")

dbutils.widgets.text("catalog", "fraud_demo")
dbutils.widgets.text("target", "dev")

catalog = dbutils.widgets.get("catalog")
target = dbutils.widgets.get("target")

# --------------------------------------------------
# Safety guard: this must never touch the prod endpoint
# unless it's actually running as the prod target
# --------------------------------------------------
if target != "prod":
    print(
        f"Target is '{target}', not 'prod' — skipping serving endpoint update.")
    dbutils.notebook.exit("skipped: not prod")

registered_model_name = f"{catalog}.models.fraud_detector"
endpoint_name = "fraud-detector-endpoint"

client = MlflowClient()
champion_mv = client.get_model_version_by_alias(
    registered_model_name, "champion")
champion_version = str(champion_mv.version)

print(f"Champion model version: {champion_version}")

w = WorkspaceClient()
endpoint = w.serving_endpoints.get(name=endpoint_name)

print(f"Endpoint found: {endpoint.name}")
print(f"Endpoint ready state: {endpoint.state.ready}")

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
