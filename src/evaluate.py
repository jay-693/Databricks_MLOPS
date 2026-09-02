# Databricks notebook source

import mlflow
from sklearn.metrics import average_precision_score, roc_auc_score

# Get information from training task
run_id = dbutils.jobs.taskValues.get(
    taskKey="train_models",
    key="run_id"
)

best_algorithm = dbutils.jobs.taskValues.get(
    taskKey="train_models",
    key="best_algorithm"
)

print(f"Evaluating: {best_algorithm}")
print(f"Run ID: {run_id}")

# Retrieve the MLflow run
run = mlflow.get_run(run_id)

print("MLflow run retrieved successfully")
print(f"Run status: {run.info.status}")

# Get logged metrics
metrics = run.data.metrics

avg_precision = metrics.get("avg_precision", 0.0)
roc_auc = metrics.get("roc_auc", 0.0)
precision_at_80recall = metrics.get(
    "precision_at_80recall",
    0.0
)

print(f"Average Precision: {avg_precision}")
print(f"ROC AUC: {roc_auc}")
print(f"Precision @ 80% Recall: {precision_at_80recall}")

# Example business threshold
eval_passed = (
    avg_precision >= 0.50
    and precision_at_80recall >= 0.20
)

print(f"Evaluation passed: {eval_passed}")

# Pass result to next task
dbutils.jobs.taskValues.set(
    key="eval_passed",
    value=eval_passed
)

dbutils.jobs.taskValues.set(
    key="run_id",
    value=run_id
)
