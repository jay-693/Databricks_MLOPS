# Databricks notebook source
import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")

dbutils.widgets.text("source_catalog", "fraud_demo_test")
dbutils.widgets.text("target_catalog", "fraud_demo_prod")

source_catalog = dbutils.widgets.get("source_catalog")
target_catalog = dbutils.widgets.get("target_catalog")

source_model_name = f"{source_catalog}.models.fraud_detector"
target_model_name = f"{target_catalog}.models.fraud_detector"

client = MlflowClient()

# --------------------------------------------------
# Get the source environment's current champion
# --------------------------------------------------
source_champion = client.get_model_version_by_alias(
    source_model_name, "champion")
source_run = client.get_run(source_champion.run_id)
source_metric = source_run.data.metrics.get("avg_precision")

print(f"Source champion: {source_model_name} v{source_champion.version} "
      f"(avg_precision={source_metric})")

# --------------------------------------------------
# Copy the model version across catalogs (no retraining)
# --------------------------------------------------
src_uri = f"models:/{source_model_name}/{source_champion.version}"

copied_mv = client.copy_model_version(
    src_model_uri=src_uri,
    dst_name=target_model_name,
)

print(f"Copied into {target_model_name} as version {copied_mv.version}")

# --------------------------------------------------
# Compare against prod's existing champion (if any) before promoting
# --------------------------------------------------
try:
    current_prod_champion = client.get_model_version_by_alias(
        target_model_name, "champion")
    prod_metric = client.get_run(
        current_prod_champion.run_id).data.metrics.get("avg_precision")
    print(
        f"Current prod champion: v{current_prod_champion.version} (avg_precision={prod_metric})")
except Exception:
    prod_metric = -1
    print("No existing prod champion found — this will become champion by default.")

promoted = source_metric is not None and source_metric > (prod_metric or -1)

if promoted:
    client.set_registered_model_alias(
        target_model_name, "champion", copied_mv.version)
    print(
        f"Promoted: prod 'champion' now points to version {copied_mv.version}")
else:
    client.set_registered_model_alias(
        target_model_name, "challenger", copied_mv.version)
    print("Not promoted — copied version registered as 'challenger' only.")

dbutils.jobs.taskValues.set(key="promoted", value=promoted)
