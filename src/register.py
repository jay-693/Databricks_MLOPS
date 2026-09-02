# Databricks notebook source

import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")

dbutils.widgets.text("catalog", "fraud_demo")

catalog = dbutils.widgets.get("catalog")

# Get the winning model's MLflow run ID
run_id = dbutils.jobs.taskValues.get(
    taskKey="train_models",
    key="run_id"
)

print(f"Registering model from run: {run_id}")

model_uri = f"runs:/{run_id}/model"
registered_model_name = f"{catalog}.models.fraud_detector"

mv = mlflow.register_model(
    model_uri=model_uri,
    name=registered_model_name
)

print(f"Registered model version: {mv.version}")

client = MlflowClient()

client.set_registered_model_alias(
    name=registered_model_name,
    alias="challenger",
    version=mv.version
)

print(f"Alias 'challenger' now points to version {mv.version}")

# --- Promotion check: does the new model beat the current champion? ---
new_metric = client.get_run(run_id).data.metrics["avg_precision"]

try:
    champion_mv = client.get_model_version_by_alias(
        registered_model_name, "champion")
    champion_metric = client.get_run(
        champion_mv.run_id).data.metrics["avg_precision"]
    print(
        f"Current champion: v{champion_mv.version}, avg_precision={champion_metric:.4f}")
except Exception:
    champion_metric = -1  # no champion alias set yet — first run always promotes
    print("No existing champion found — this run will become champion by default.")

print(f"Challenger: v{mv.version}, avg_precision={new_metric:.4f}")

promoted = new_metric > champion_metric
if promoted:
    client.set_registered_model_alias(
        registered_model_name, "champion", mv.version)
    print(f"Promoted: 'champion' now points to version {mv.version}")
else:
    print("Not promoted — challenger did not beat current champion.")

dbutils.jobs.taskValues.set(key="promoted", value=promoted)
