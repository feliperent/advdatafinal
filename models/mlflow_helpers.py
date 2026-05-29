# MLflow helpers: experiment setup + run context manager.
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import mlflow

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file://{REPO_ROOT / 'mlruns'}")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("advdatafinal")

@contextmanager
def run(rung: int, fold_id: str, model_family: str, n_features: int):
    with mlflow.start_run(run_name=f"rung{rung}_{fold_id}_{model_family}"):
        mlflow.log_param("rung", rung)
        mlflow.log_param("fold_id", fold_id)
        mlflow.log_param("model_family", model_family)
        mlflow.log_param("n_features", n_features)
        yield mlflow
