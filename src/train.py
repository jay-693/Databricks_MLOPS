# Databricks notebook source

import numpy as np
import pandas as pd
import yaml
import mlflow
import mlflow.pyfunc
import sklearn

from mlflow.models import infer_signature

from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.model_selection import (
    ParameterGrid,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

mlflow.set_experiment(
    "/Workspace/Users/vattikutivijay693@gmail.com/fraud_mlops"
)

# Models are logged explicitly below, with a signature.
mlflow.autolog(log_models=False)

# --------------------------------------------------
# Parameters (notebook widgets / job base_parameters)
# --------------------------------------------------

dbutils.widgets.text("catalog", "fraud_demo")

# Recall the decision threshold is tuned to reach (on the validation slice)
dbutils.widgets.text("target_recall", "0.80")

# How the winning algorithm is chosen (measured on the validation slice):
#   avg_precision | precision_at_80recall | precision_at_target_recall
dbutils.widgets.text("selection_metric", "avg_precision")

# true  -> log a model that applies the tuned threshold (served output = 0/1 at that threshold)
# false -> log the plain sklearn model (predict() = default 0.5 cutoff), as before
dbutils.widgets.text("use_threshold_wrapper", "true")

# Also keep the raw sklearn model next to the wrapper (doubles artifact size)
dbutils.widgets.text("log_raw_sklearn_model", "false")

# Hyperparameter search: all frauds + this many legit rows per fraud
dbutils.widgets.text("search_neg_per_pos", "40")
dbutils.widgets.text("search_n_iter", "8")
dbutils.widgets.text("search_cv", "3")

# Share of the training rows held out to choose the threshold / the winner
dbutils.widgets.text("val_fraction", "0.10")

# Warn when the chosen threshold would raise more than this many alerts per true fraud
# (a target_recall set beyond the knee of the precision-recall curve flags huge volumes)
dbutils.widgets.text("max_alerts_per_fraud", "100")


def _as_bool(x):
    return str(x).strip().lower() in ("1", "true", "yes", "y")


catalog = dbutils.widgets.get("catalog")
TARGET_RECALL = float(dbutils.widgets.get("target_recall"))
SELECTION_METRIC = dbutils.widgets.get("selection_metric").strip()
USE_THRESHOLD_WRAPPER = _as_bool(dbutils.widgets.get("use_threshold_wrapper"))
LOG_RAW_SKLEARN = _as_bool(dbutils.widgets.get("log_raw_sklearn_model"))
SEARCH_NEG_PER_POS = int(dbutils.widgets.get("search_neg_per_pos"))
SEARCH_N_ITER = int(dbutils.widgets.get("search_n_iter"))
SEARCH_CV = int(dbutils.widgets.get("search_cv"))
VAL_FRACTION = float(dbutils.widgets.get("val_fraction"))
MAX_ALERTS_PER_FRAUD = float(dbutils.widgets.get("max_alerts_per_fraud"))

SELECTION_KEYS = {
    "avg_precision": "val_avg_precision",
    "precision_at_80recall": "val_precision_at_80recall",
    "precision_at_target_recall": "val_precision_at_target_recall",
}

assert 0.0 < TARGET_RECALL < 1.0, "target_recall must be between 0 and 1"
assert SELECTION_METRIC in SELECTION_KEYS, (
    f"selection_metric must be one of {list(SELECTION_KEYS)}"
)
assert 0.0 < VAL_FRACTION < 0.5, "val_fraction must be between 0 and 0.5"

# --------------------------------------------------
# Load data
# --------------------------------------------------

df = spark.table(
    f"{catalog}.silver.fraud_features"
).toPandas()

NON_FEATURE_COLS = [
    "isFraud",
    "txn_id",
    "ingest_ts",
    "batch_id"
]

feature_cols = [
    c for c in df.columns
    if c not in NON_FEATURE_COLS
]

X = df[feature_cols]
y = df["isFraud"]

n_null_cells = int(X.isna().sum().sum())
if n_null_cells:
    print(f"WARNING: {n_null_cells} null cells in the feature table")

print(
    f"Rows: {len(X):,} | features: {len(feature_cols)} | "
    f"frauds: {int(y.sum()):,} ({100 * y.mean():.4f}%)"
)

# --------------------------------------------------
# Splits
#   test  : final, untouched evaluation
#   val   : chooses the decision threshold and the winning algorithm
#   fit   : trains the models (the search sample is drawn from here)
# --------------------------------------------------

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

X_fit, X_val, y_fit, y_val = train_test_split(
    X_trainval,
    y_trainval,
    test_size=VAL_FRACTION,
    stratify=y_trainval,
    random_state=42
)

print(
    f"fit: {len(X_fit):,} rows ({int(y_fit.sum()):,} frauds) | "
    f"val: {len(X_val):,} ({int(y_val.sum()):,}) | "
    f"test: {len(X_test):,} ({int(y_test.sum()):,})"
)

# --------------------------------------------------
# Hyperparameter search sample:
# keep EVERY fraud and add a multiple of legit rows.
# (A plain random sample of a 0.1%-fraud table holds
#  only ~100 frauds - far too few to rank settings.)
# --------------------------------------------------

pos_idx = y_fit[y_fit == 1].index
neg_pool = y_fit[y_fit == 0]
neg_idx = neg_pool.sample(
    n=min(len(neg_pool), SEARCH_NEG_PER_POS * len(pos_idx)),
    random_state=42
).index

search_idx = pos_idx.union(neg_idx)
X_search = X_fit.loc[search_idx]
y_search = y_fit.loc[search_idx]

print(
    f"Searching hyperparameters on {len(X_search):,} rows "
    f"({int(y_search.sum()):,} frauds, {100 * y_search.mean():.2f}%) "
    f"- fit set has {len(X_fit):,} rows"
)

# --------------------------------------------------
# Models to compare.
# Only algorithms that appear in conf/model_params.yml are run,
# so remove a block from the YAML to skip that algorithm.
# --------------------------------------------------

with open("../conf/model_params.yml") as f:
    param_grids = yaml.safe_load(f)

MODEL_FACTORIES = {

    # scaled, because logistic regression is scale-sensitive
    "logistic_regression":
        lambda: Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(random_state=42)),
        ]),

    "random_forest":
        lambda: RandomForestClassifier(
            n_jobs=-1,
            random_state=42
        ),

    "gradient_boosting":
        lambda: GradientBoostingClassifier(
            random_state=42
        ),

    # fast on millions of rows; class_weight needs scikit-learn >= 1.2
    "hist_gradient_boosting":
        lambda: HistGradientBoostingClassifier(
            random_state=42
        ),
}

unknown = [k for k in param_grids if k not in MODEL_FACTORIES]
if unknown:
    print(f"WARNING: ignoring unknown models in model_params.yml: {unknown}")

models_to_run = [k for k in param_grids if k in MODEL_FACTORIES]
assert models_to_run, "No known models found in conf/model_params.yml"
print(f"Models in this bake-off: {models_to_run}")


def search_space(model, grid):
    # Pipeline parameters must be addressed as <step>__<param>
    if isinstance(model, Pipeline):
        return {f"clf__{k}": v for k, v in grid.items()}
    return grid


# --------------------------------------------------
# Metric helpers
# --------------------------------------------------

def precision_at_recall(y_true, probs, min_recall):
    precisions, recalls, _ = precision_recall_curve(y_true, probs)
    mask = recalls >= min_recall
    return float(precisions[mask].max()) if mask.any() else 0.0


def choose_threshold(y_true, probs, target_recall):
    """Highest threshold whose recall is still >= target_recall."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    ok = np.where(recalls[:-1] >= target_recall)[0]
    if len(ok) == 0:
        raise ValueError("No threshold reaches the target recall")
    i = ok[-1]
    return float(thresholds[i]), float(precisions[i]), float(recalls[i])


def operating_points(y_true, probs, targets=(0.5, 0.6, 0.7, 0.8, 0.9)):
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    rows = []
    for r in targets:
        ok = np.where(recalls[:-1] >= r)[0]
        if len(ok):
            i = ok[-1]
            rows.append({
                "target_recall": r,
                "threshold": float(thresholds[i]),
                "precision": float(precisions[i]),
                "alerts_per_fraud": float(1 / max(precisions[i], 1e-12)),
            })
    return pd.DataFrame(rows)


def confusion_at(y_true, probs, threshold):
    yt = np.asarray(y_true).astype(bool)
    pred = np.asarray(probs) >= threshold
    tp = int((pred & yt).sum())
    fp = int((pred & ~yt).sum())
    fn = int((~pred & yt).sum())
    tn = int((~pred & ~yt).sum())
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "alerts_per_fraud": (tp + fp) / tp if tp else 0.0,
    }


class ThresholdedModel(mlflow.pyfunc.PythonModel):
    """Serves 1 when P(fraud) >= threshold, else 0."""

    def __init__(self, model, threshold):
        self.model = model
        self.threshold = float(threshold)

    def predict(self, context, model_input, params=None):
        proba = self.model.predict_proba(model_input)[:, 1]
        return (proba >= self.threshold).astype(int)


results = []

with mlflow.start_run(
    run_name="model_bakeoff"
) as parent_run:

    mlflow.log_params({
        "target_recall": TARGET_RECALL,
        "selection_metric": SELECTION_METRIC,
        "use_threshold_wrapper": USE_THRESHOLD_WRAPPER,
        "val_fraction": VAL_FRACTION,
        "search_neg_per_pos": SEARCH_NEG_PER_POS,
    })

    for name in models_to_run:

        base_model = MODEL_FACTORIES[name]()

        with mlflow.start_run(
            run_name=name,
            nested=True
        ) as child_run:

            # --------------------------------------
            # Step 1: Hyperparameter search
            #         (on the fraud-rich search sample)
            # --------------------------------------

            space = search_space(base_model, param_grids[name])
            n_iter = min(SEARCH_N_ITER, len(ParameterGrid(space)))

            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=space,
                n_iter=n_iter,
                scoring="average_precision",
                cv=StratifiedKFold(
                    n_splits=SEARCH_CV,
                    shuffle=True,
                    random_state=42
                ),
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
            # Step 2: Train the final model on the
            #         fit split (all non-held-out rows)
            # --------------------------------------

            final_model = clone(base_model).set_params(**best_params)

            final_model.fit(
                X_fit,
                y_fit
            )

            # --------------------------------------
            # Step 3: Validation slice -> choose the
            #         decision threshold for the target recall
            # --------------------------------------

            val_probs = final_model.predict_proba(X_val)[:, 1]

            val_ap = average_precision_score(y_val, val_probs)
            val_p80 = precision_at_recall(y_val, val_probs, 0.80)

            threshold, val_precision_at_target, val_recall_at_threshold = (
                choose_threshold(y_val, val_probs, TARGET_RECALL)
            )

            # --------------------------------------
            # Step 4: Test split -> final, untouched metrics
            # --------------------------------------

            probs = final_model.predict_proba(X_test)[:, 1]

            ap = average_precision_score(y_test, probs)
            auc = roc_auc_score(y_test, probs)
            precision_at_80recall = precision_at_recall(y_test, probs, 0.80)

            op = operating_points(y_test, probs)
            cm = confusion_at(y_test, probs, threshold)

            print(
                f"{name}: test AP={ap:.4f} | threshold={threshold:.5f} "
                f"-> recall={cm['recall']:.3f}, precision={cm['precision']:.3f}, "
                f"{cm['alerts_per_fraud']:.1f} alerts per fraud"
            )

            flagged_rate = (cm["tp"] + cm["fp"]) / len(y_test)

            if cm["alerts_per_fraud"] > MAX_ALERTS_PER_FRAUD:
                print(
                    f"WARNING ({name}): threshold {threshold:.5f} implies "
                    f"{cm['alerts_per_fraud']:.0f} alerts per fraud "
                    f"({100 * flagged_rate:.2f}% of transactions flagged) - above the "
                    f"limit of {MAX_ALERTS_PER_FRAUD:.0f}. target_recall={TARGET_RECALL} is "
                    "probably past the knee of the PR curve; see operating_points.csv."
                )

            # --------------------------------------
            # Log params / metrics / artifacts
            # --------------------------------------

            mlflow.log_params(best_params)
            mlflow.log_param("decision_threshold", threshold)

            metrics = {
                # names used by evaluate.py / register.py (test split)
                "avg_precision": float(ap),
                "roc_auc": float(auc),
                "precision_at_80recall": float(precision_at_80recall),
                "cv_best_score_on_sample": float(search.best_score_),

                # validation slice (selection + threshold)
                "val_avg_precision": float(val_ap),
                "val_precision_at_80recall": float(val_p80),
                "val_precision_at_target_recall": float(val_precision_at_target),
                "val_recall_at_threshold": float(val_recall_at_threshold),

                # test split at the chosen threshold
                "decision_threshold": float(threshold),
                "recall_at_threshold": float(cm["recall"]),
                "precision_at_threshold": float(cm["precision"]),
                "alerts_per_fraud_at_threshold": float(cm["alerts_per_fraud"]),
                "flagged_rate_at_threshold": float(flagged_rate),
                "tp_at_threshold": float(cm["tp"]),
                "fp_at_threshold": float(cm["fp"]),
                "fn_at_threshold": float(cm["fn"]),
            }
            for r in op.itertuples():
                metrics[f"precision_at_recall_{int(round(r.target_recall * 100))}"] = float(
                    r.precision)

            mlflow.log_metrics(metrics)

            mlflow.log_text(
                op.to_csv(index=False),
                "operating_points.csv"
            )

            mlflow.set_tag(
                "algorithm",
                name
            )

            # --------------------------------------
            # IMPORTANT:
            # Create MLflow model signature
            # --------------------------------------

            sample = X_fit.head(1000)

            if USE_THRESHOLD_WRAPPER:

                served_model = ThresholdedModel(final_model, threshold)

                signature = infer_signature(
                    sample,
                    served_model.predict(None, sample)
                )

                mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=served_model,
                    signature=signature,
                    input_example=sample.head(5),
                    extra_pip_requirements=[
                        f"scikit-learn=={sklearn.__version__}"
                    ],
                )

                if LOG_RAW_SKLEARN:
                    mlflow.sklearn.log_model(
                        sk_model=final_model,
                        artifact_path="sklearn_model",
                        signature=infer_signature(
                            sample,
                            final_model.predict(sample)
                        ),
                        input_example=sample.head(5),
                    )

            else:

                signature = infer_signature(
                    sample,
                    final_model.predict(sample)
                )

                mlflow.sklearn.log_model(
                    sk_model=final_model,
                    artifact_path="model",
                    signature=signature,
                    input_example=sample.head(5)
                )

            # --------------------------------------
            # Store results
            # --------------------------------------

            results.append({

                "algorithm": name,

                "run_id":
                    child_run.info.run_id,

                # test split (downstream gate / registry)
                "avg_precision":
                    float(ap),

                "precision_at_80recall":
                    float(precision_at_80recall),

                "decision_threshold":
                    float(threshold),

                "recall_at_threshold":
                    float(cm["recall"]),

                "precision_at_threshold":
                    float(cm["precision"]),

                "alerts_per_fraud_at_threshold":
                    float(cm["alerts_per_fraud"]),

                # validation slice (used to pick the winner)
                "val_avg_precision":
                    float(val_ap),

                "val_precision_at_80recall":
                    float(val_p80),

                "val_precision_at_target_recall":
                    float(val_precision_at_target),
            })

    # ------------------------------------------
    # Select best model - on the VALIDATION slice,
    # so the test split stays untouched
    # ------------------------------------------

    selection_key = SELECTION_KEYS[SELECTION_METRIC]

    best = sorted(
        results,
        key=lambda r: (
            r[selection_key],
            r["val_avg_precision"]
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
# Summary
# ----------------------------------------------

summary = pd.DataFrame(results).sort_values(selection_key, ascending=False)
print(summary.round(4).to_string(index=False))

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

dbutils.jobs.taskValues.set(
    key="decision_threshold",
    value=float(best["decision_threshold"])
)

dbutils.jobs.taskValues.set(
    key="recall_at_threshold",
    value=float(best["recall_at_threshold"])
)

print(
    f"Best algorithm: {best['algorithm']}"
)

print(
    f"Best MLflow run ID: {run_id}"
)

print(
    f"Decision threshold: {best['decision_threshold']:.5f} "
    f"(test recall {best['recall_at_threshold']:.3f}, "
    f"precision {best['precision_at_threshold']:.3f}, "
    f"{best['alerts_per_fraud_at_threshold']:.1f} alerts per fraud)"
)
