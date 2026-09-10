# Databricks notebook source

import mlflow
import yaml

from mlflow.models import infer_signature

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve
)

mlflow.set_experiment(
    "/Workspace/Users/vattikutivijay693@gmail.com/fraud_mlops"
)

# Don't let autologging create the model artifact.
# We will log it explicitly with a signature.
mlflow.autolog(log_models=False)

dbutils.widgets.text("catalog", "fraud_demo")
catalog = dbutils.widgets.get("catalog")

df = spark.table(
    f"{catalog}.silver.txn_features"
).toPandas()

NON_FEATURE_COLS = [
    "Class",
    "txn_id",
    "ingest_ts",
    "batch_id"
]

feature_cols = [
    c for c in df.columns
    if c not in NON_FEATURE_COLS
]

X = df[feature_cols]
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# --------------------------------------------------
# Hyperparameter search sample
# --------------------------------------------------

SEARCH_SAMPLE_SIZE = 40000

if len(X_train) > SEARCH_SAMPLE_SIZE:

    X_search, _, y_search, _ = train_test_split(
        X_train,
        y_train,
        train_size=SEARCH_SAMPLE_SIZE,
        stratify=y_train,
        random_state=42,
    )

else:

    X_search = X_train
    y_search = y_train

print(
    f"Searching hyperparameters on {len(X_search)} rows "
    f"(full train set: {len(X_train)})"
)

with open("../conf/model_params.yml") as f:
    param_grids = yaml.safe_load(f)

base_models = {

    "logistic_regression":
        LogisticRegression(random_state=42),

    "random_forest":
        RandomForestClassifier(
            n_jobs=-1,
            random_state=42
        ),

    "gradient_boosting":
        GradientBoostingClassifier(
            random_state=42
        ),
}

results = []

with mlflow.start_run(
    run_name="model_bakeoff"
) as parent_run:

    for name, base_model in base_models.items():

        with mlflow.start_run(
            run_name=name,
            nested=True
        ) as child_run:

            # --------------------------------------
            # Step 1: Hyperparameter search
            # --------------------------------------

            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_grids[name],
                n_iter=4,
                scoring="average_precision",
                cv=2,
                random_state=42,
                n_jobs=-1,
            )

            search.fit(
                X_search,
                y_search
            )

            best_params = search.best_params_

            print(
                f"{name} best params: "
                f"{best_params}"
            )

            # --------------------------------------
            # Step 2: Train final model on FULL data
            # --------------------------------------

            final_model = base_model.__class__(
                **{
                    **base_model.get_params(),
                    **best_params
                }
            )

            final_model.fit(
                X_train,
                y_train
            )

            # --------------------------------------
            # Predictions
            # --------------------------------------

            probs = final_model.predict_proba(
                X_test
            )[:, 1]

            ap = average_precision_score(
                y_test,
                probs
            )

            auc = roc_auc_score(
                y_test,
                probs
            )

            precisions, recalls, _ = precision_recall_curve(
                y_test,
                probs
            )

            mask = recalls >= 0.80

            precision_at_80recall = (
                precisions[mask].max()
                if mask.any()
                else 0.0
            )

            # --------------------------------------
            # Log metrics
            # --------------------------------------

            mlflow.log_params(
                best_params
            )

            mlflow.log_metrics({
                "avg_precision": float(ap),

                "roc_auc": float(auc),

                "precision_at_80recall":
                    float(precision_at_80recall),

                "cv_best_score_on_sample":
                    float(search.best_score_),
            })

            mlflow.set_tag(
                "algorithm",
                name
            )

            # --------------------------------------
            # IMPORTANT:
            # Create MLflow model signature
            # --------------------------------------

            signature = infer_signature(
                X_train,
                final_model.predict(X_train)
            )

            # --------------------------------------
            # Log the FULL-data model
            # WITH SIGNATURE
            # --------------------------------------

            mlflow.sklearn.log_model(
                sk_model=final_model,
                artifact_path="model",
                signature=signature,
                input_example=X_train.head(5)
            )

            # --------------------------------------
            # Store results
            # --------------------------------------

            results.append({

                "algorithm": name,

                "run_id":
                    child_run.info.run_id,

                "avg_precision":
                    float(ap),

                "precision_at_80recall":
                    float(precision_at_80recall),
            })

    # ------------------------------------------
    # Select best model
    # ------------------------------------------

    best = sorted(
        results,
        key=lambda r: (
            r["avg_precision"],
            r["precision_at_80recall"]
        ),
        reverse=True,
    )[0]

    mlflow.log_param(
        "best_algorithm",
        best["algorithm"]
    )

    mlflow.set_tag(
        "best_run_id",
        best["run_id"]
    )

    run_id = best["run_id"]

    parent_run_id = parent_run.info.run_id


# ----------------------------------------------
# Pass values to downstream tasks
# ----------------------------------------------

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

print(
    f"Best algorithm: {best['algorithm']}"
)

print(
    f"Best MLflow run ID: {run_id}"
)