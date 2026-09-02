# Databricks notebook source
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

mlflow.set_experiment(
    "/Workspace/Users/vattikutivijay693@gmail.com/fraud_mlops")
mlflow.autolog(log_models=True)

dbutils.widgets.text("catalog", "fraud_demo")

catalog = dbutils.widgets.get("catalog")

df = spark.table(f"{catalog}.silver.txn_features").toPandas()
NON_FEATURE_COLS = ["Class", "txn_id", "ingest_ts", "batch_id"]
feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
X = df[feature_cols]
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# imbalance-aware weighting so all 3 models get a fair shot at the minority class
candidates = {
    "logistic_regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42),
    "random_forest": RandomForestClassifier(
        n_estimators=300, max_depth=12, class_weight="balanced",
        n_jobs=-1, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42),
}

results = []

with mlflow.start_run(run_name="model_bakeoff") as parent_run:
    for name, model in candidates.items():
        with mlflow.start_run(run_name=name, nested=True) as child_run:
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]

            # primary metric: imbalance-aware
            ap = average_precision_score(y_test, probs)
            auc = roc_auc_score(y_test, probs)

            # precision at a fixed, business-relevant recall (catch >=80% of fraud)
            precisions, recalls, _ = precision_recall_curve(y_test, probs)
            mask = recalls >= 0.80
            precision_at_80recall = precisions[mask].max(
            ) if mask.any() else 0.0

            mlflow.log_metrics({
                "avg_precision": ap,
                "roc_auc": auc,
                "precision_at_80recall": precision_at_80recall,
            })
            mlflow.set_tag("algorithm", name)

            results.append({
                "algorithm": name,
                "run_id": child_run.info.run_id,
                "avg_precision": ap,
                "roc_auc": auc,
                "precision_at_80recall": precision_at_80recall,
            })

    # pick the winner: rank by avg_precision (imbalance-aware), tie-break on precision_at_80recall
    best = sorted(
        results,
        key=lambda r: (r["avg_precision"], r["precision_at_80recall"]),
        reverse=True,
    )[0]

    mlflow.log_param("best_algorithm", best["algorithm"])
mlflow.set_tag("best_run_id", best["run_id"])

run_id = best["run_id"]
parent_run_id = parent_run.info.run_id

# Pass values to downstream Databricks Job tasks
dbutils.jobs.taskValues.set(
    key="run_id",
    value=run_id
)

dbutils.jobs.taskValues.set(
    key="parent_run_id",
    value=parent_run_id
)

dbutils.jobs.taskValues.set(
    key="best_algorithm",
    value=best["algorithm"]
)

dbutils.jobs.taskValues.set(
    key="avg_precision",
    value=float(best["avg_precision"])
)

print(f"Best algorithm: {best['algorithm']}")
print(f"Best MLflow run ID: {run_id}")
print(f"Parent MLflow run ID: {parent_run_id}")
