"""
train.py
========
Model training, tuning, evaluation, and MLflow experiment-tracking script.
Covers Assignment Task 2 (Feature Engineering & Model Development), Task 3
(Experiment Tracking), and produces the artifact consumed by Task 4 (Model
Packaging & Reproducibility).

What this script does, end to end
----------------------------------
1. Loads the cleaned dataset (data/processed/heart_disease_clean.csv) --
   produced by src/download_data.py.
2. Splits into train/test sets (stratified, so the ~54/46 class balance is
   preserved in both splits).
3. For each of three candidate models -- Logistic Regression, Random Forest,
   and XGBoost -- builds a full sklearn Pipeline (preprocessing + estimator,
   see src/features.py), runs GridSearchCV (5-fold stratified cross-
   validation) to tune hyperparameters, and evaluates the BEST estimator on
   the held-out test set using accuracy, precision, recall, F1, and ROC-AUC.
4. Logs every run to MLflow: hyperparameters, CV results, test metrics, a
   confusion-matrix plot, an ROC-curve plot, and the fitted pipeline itself
   as an MLflow model artifact.
5. Picks the model with the highest test ROC-AUC as the "final" model and
   additionally saves it as a plain joblib file
   (models/heart_disease_pipeline.joblib) -- this is the artifact the
   FastAPI service loads at inference time, independent of a running MLflow
   server (important for Docker/K8s deployment where we don't want a hard
   dependency on an MLflow tracking server being reachable).

Why GridSearchCV instead of manual tuning?
    GridSearchCV performs an exhaustive search over a hyperparameter grid
    WITH cross-validation baked in, so the reported CV score already
    accounts for variance across folds -- this is what the assignment asks
    for ("Evaluate using cross-validation") and is more defensible than a
    single train/validation split for a dataset this small (303 rows).

Usage
-----
    python src/train.py
"""

import os  # noqa: E402
import sys  # noqa: E402
import json  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")  # headless backend -- required in Docker/CI (no display)
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

# Local imports -- repo root is added to sys.path so this script runs whether
# invoked as `python src/train.py` (from repo root) or from inside src/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import build_full_pipeline, split_features_target  # noqa: E402

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join("data", "processed", "heart_disease_clean.csv")
MODELS_DIR = "models"
PLOTS_DIR = os.path.join("reports", "figures")
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "heart_disease_pipeline.joblib")
METRICS_SUMMARY_PATH = os.path.join(MODELS_DIR, "metrics_summary.json")

RANDOM_STATE = 42  # fixed seed everywhere reproducibility matters (Task 4)
TEST_SIZE = 0.2
CV_FOLDS = 5

MLFLOW_EXPERIMENT_NAME = "heart-disease-classification"

# MLflow tracking backend: defaults to a local SQLite database file
# (mlflow.db in the repo root). SQLite is used instead of the legacy plain
# "./mlruns" file-store because recent MLflow versions (>=2.17) put the
# file-store backend into maintenance mode. SQLite requires zero external
# services (no Postgres/MySQL container needed) yet still gives full MLflow
# UI functionality (`mlflow ui --backend-store-uri sqlite:///mlflow.db`),
# which keeps local setup fast -- important given the tight assignment
# timeline. Override with the MLFLOW_TRACKING_URI env var to point at a
# remote MLflow server instead.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the cleaned dataset produced by src/download_data.py.

    Args:
        path (str): Path to the cleaned CSV.

    Returns:
        pd.DataFrame: The cleaned heart-disease dataframe.

    Raises:
        FileNotFoundError: with a helpful message if the cleaned CSV has not
            been generated yet (tells the user to run download_data.py first)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cleaned dataset not found at '{path}'. "
            "Run `python src/download_data.py` first to download and clean the data."
        )
    return pd.read_csv(path)


def get_model_grid():
    """
    Define the candidate models and their GridSearchCV hyperparameter grids.

    Design notes:
        - Logistic Regression: tuned over regularisation strength C and
          penalty type. It is our fast, highly-interpretable BASELINE model
          -- clinicians and stakeholders can reason about its coefficients.
        - Random Forest: tuned over tree count, depth, and split criteria.
          Captures non-linear feature interactions the linear baseline
          cannot (e.g. interaction between age and max heart rate).
        - XGBoost: tuned over tree count, depth, and learning rate. Included
          as a third, typically stronger, gradient-boosted baseline (the
          assignment explicitly lists XGBoost as an acceptable model choice
          and having 3 models gives a more convincing model-selection
          narrative for the report than the minimum of 2).

    Returns:
        dict: mapping model_name -> (estimator, param_grid) where param_grid
            keys are prefixed "classifier__" to target the estimator step
            inside the full sklearn Pipeline (see
            src/features.build_full_pipeline).
    """
    grids = {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__penalty": ["l2"],
                "classifier__solver": ["lbfgs"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [3, 5, 8, None],
                "classifier__min_samples_leaf": [1, 2, 4],
            },
        ),
        "xgboost": (
            XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
            ),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [2, 3, 4],
                "classifier__learning_rate": [0.01, 0.05, 0.1],
            },
        ),
    }
    return grids


def evaluate_model(pipeline, X_test, y_test) -> dict:
    """
    Compute the full evaluation-metric suite the assignment asks for
    (accuracy, precision, recall, F1, ROC-AUC) on a held-out test set.

    Args:
        pipeline (sklearn.pipeline.Pipeline): fitted end-to-end pipeline.
        X_test (pd.DataFrame): held-out feature matrix.
        y_test (pd.Series): held-out true labels.

    Returns:
        tuple(dict, array, array): (metrics dict, y_pred, y_proba)
    """
    y_pred = pipeline.predict(X_test)
    # predict_proba(...)[:, 1] = predicted probability of the POSITIVE class
    # (target == 1, i.e. disease present) -- required for ROC-AUC, which is
    # a threshold-independent metric computed from probability scores.
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    return metrics, y_pred, y_proba


def plot_confusion_matrix(y_test, y_pred, model_name: str, out_dir: str) -> str:
    """
    Render and save a confusion-matrix heatmap for one model's test-set
    predictions. Confusion matrices are especially important in a clinical
    classifier because False Negatives (predicting "no disease" when disease
    is actually present) carry a much higher real-world cost than False
    Positives -- the plot lets a reviewer see that tradeoff at a glance.

    Args:
        y_test (array-like): true labels.
        y_pred (array-like): predicted labels.
        model_name (str): used in the plot title and output filename.
        out_dir (str): directory to save the PNG into.

    Returns:
        str: path to the saved PNG file.
    """
    os.makedirs(out_dir, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disease", "Disease"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix - " + model_name)
    fig.tight_layout()
    path = os.path.join(out_dir, f"confusion_matrix_{model_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_roc_curve(pipeline, X_test, y_test, model_name: str, out_dir: str) -> str:
    """
    Render and save an ROC curve for one model's test-set probability
    predictions. Logged to MLflow as a run artifact (Task 3 requirement:
    "Log parameters, metrics, artifacts, and plots for all runs").

    Args:
        pipeline (sklearn.pipeline.Pipeline): fitted pipeline (needs
            predict_proba).
        X_test (pd.DataFrame): held-out feature matrix.
        y_test (pd.Series): held-out true labels.
        model_name (str): used in the plot title and output filename.
        out_dir (str): directory to save the PNG into.

    Returns:
        str: path to the saved PNG file.
    """
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
    ax.set_title("ROC Curve - " + model_name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    fig.tight_layout()
    path = os.path.join(out_dir, f"roc_curve_{model_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def train_and_track_all_models():
    """
    Main orchestration function: for every candidate model in
    get_model_grid(), run GridSearchCV + evaluation, log everything to
    MLflow as a separate run, then compare all models and persist the best
    one to disk as the "final" reusable model artifact.

    Returns:
        dict: summary of all runs (model_name -> metrics dict), plus the
            name of the selected best model. Also written to
            models/metrics_summary.json for easy inspection without
            opening the MLflow UI.
    """
    df = load_data()
    X, y = split_features_target(df)

    # Stratified split: stratify=y ensures the train and test sets each keep
    # approximately the same 54%/46% class balance as the full dataset --
    # critical on a small dataset (303 rows) where a random split could
    # otherwise skew one split toward a single class.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    cv_strategy = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    all_results = {}
    fitted_pipelines = {}

    for model_name, (estimator, param_grid) in get_model_grid().items():
        print(f"\n[train] === Training {model_name} ===")
        pipeline = build_full_pipeline(estimator)

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv_strategy,
            scoring="roc_auc",  # optimise for ROC-AUC -- robust to class imbalance
            n_jobs=-1,
            refit=True,
        )

        with mlflow.start_run(run_name=model_name):
            grid_search.fit(X_train, y_train)
            best_pipeline = grid_search.best_estimator_

            metrics, y_pred, y_proba = evaluate_model(best_pipeline, X_test, y_test)
            cv_best_score = grid_search.best_score_

            # --- MLflow logging (Task 3) ---------------------------------
            # 1. Log the winning hyperparameters (strip the "classifier__"
            #    prefix for readability in the MLflow UI).
            clean_params = {
                k.replace("classifier__", ""): v
                for k, v in grid_search.best_params_.items()
            }
            mlflow.log_params(clean_params)
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("cv_folds", CV_FOLDS)
            mlflow.log_param("test_size", TEST_SIZE)

            # 2. Log metrics: CV score + all held-out test metrics.
            mlflow.log_metric("cv_best_roc_auc", cv_best_score)
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"test_{metric_name}", value)

            # 3. Log plots as artifacts.
            cm_path = plot_confusion_matrix(y_test, y_pred, model_name, PLOTS_DIR)
            roc_path = plot_roc_curve(best_pipeline, X_test, y_test, model_name, PLOTS_DIR)
            mlflow.log_artifact(cm_path, artifact_path="plots")
            mlflow.log_artifact(roc_path, artifact_path="plots")

            # 4. Log the fitted pipeline itself as an MLflow model (enables
            #    "mlflow models serve" and full lineage tracking).
            mlflow.sklearn.log_model(
                best_pipeline,
                artifact_path="model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
            )

            print(f"[train] {model_name} CV ROC-AUC: {cv_best_score:.4f} | "
                  f"Test ROC-AUC: {metrics['roc_auc']:.4f} | "
                  f"Test Accuracy: {metrics['accuracy']:.4f}")

            all_results[model_name] = {
                "cv_best_roc_auc": cv_best_score,
                **metrics,
                "best_params": clean_params,
            }
            fitted_pipelines[model_name] = best_pipeline

    # -----------------------------------------------------------------
    # Model selection: pick the model with the highest TEST ROC-AUC.
    # (We select on the held-out test set, not the CV score, because the
    # test set is the closest proxy we have to unseen production data.)
    # -----------------------------------------------------------------
    best_model_name = max(all_results, key=lambda name: all_results[name]["roc_auc"])
    best_pipeline = fitted_pipelines[best_model_name]

    best_auc = all_results[best_model_name]["roc_auc"]
    print(f"\n[train] === Best model: {best_model_name} (Test ROC-AUC = {best_auc:.4f}) ===")

    # Persist the final chosen pipeline (preprocessing + model) as a single
    # joblib artifact -- this is what the FastAPI service loads.
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_pipeline, FINAL_MODEL_PATH)
    print(f"[train] Saved final model pipeline to {FINAL_MODEL_PATH}")

    summary = {
        "best_model": best_model_name,
        "results": all_results,
    }
    with open(METRICS_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[train] Saved metrics summary to {METRICS_SUMMARY_PATH}")

    return summary


if __name__ == "__main__":
    train_and_track_all_models()
