# Rung 0: per-stock ARIMA on log_ret_1d.
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from pmdarima import auto_arima
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

from ingest.common import pg_conn
from models.mlflow_helpers import run as mlflow_run
from models.walkforward import folds

warnings.filterwarnings("ignore")

def predict_5d_direction(log_returns: pd.Series) -> tuple[float, int]:
    """Fit ARIMA on log_returns, predict next 5 daily log-returns, sum, return (prob_up, class)."""
    series = pd.Series(log_returns).dropna()
    if len(series) < 60:
        return 0.5, 0
    try:
        model = auto_arima(
            series,
            suppress_warnings=True,
            error_action="ignore",
            max_p=2,
            max_d=1,
            max_q=2,
            seasonal=False,
            with_intercept=False,
        )
        forecast = model.predict(n_periods=5)
        cum = float(np.sum(forecast))
        # Normalise by train-period 5-day vol so prob is comparable across stocks
        sigma_5d = float(series.tail(60).std()) * math.sqrt(5)
        z = cum / max(sigma_5d, 1e-6)
        prob_up = 1.0 / (1.0 + math.exp(-z))
        return prob_up, int(prob_up > 0.5)
    except Exception:
        return 0.5, 0

def load_panel() -> pd.DataFrame:
    with pg_conn() as conn:
        return pd.read_sql(
            """
            SELECT trade_date::date, symbol, company_key, log_ret_1d, y_5d_up
            FROM gold.fct_feature_panel_daily
            WHERE y_5d_up IS NOT NULL AND log_ret_1d IS NOT NULL
            ORDER BY symbol, trade_date
            """,
            conn,
        )

def write_predictions(rows: list[tuple]) -> None:
    if not rows:
        return
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS gold.fct_predictions (
              prediction_key bigserial PRIMARY KEY,
              date_key       text NOT NULL,
              company_key    text NOT NULL,
              model_rung     smallint NOT NULL,
              fold_id        text NOT NULL,
              prob_up        numeric(8,5) NOT NULL,
              predicted_class smallint NOT NULL,
              shap_json      jsonb
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS ix_pred_dcr ON gold.fct_predictions (date_key, company_key, model_rung)")
        cur.executemany(
            """
            INSERT INTO gold.fct_predictions
              (date_key, company_key, model_rung, fold_id, prob_up, predicted_class)
            VALUES (md5(%s), %s, %s, %s, %s, %s)
            """,
            rows,
        )

def main(max_test_per_stock_per_fold: int = 20) -> None:
    """Fit per stock per fold. To keep wall time reasonable we sample up to N test rows per (stock, fold)."""
    df = load_panel()
    print(f"Loaded {len(df)} panel rows for {df['symbol'].nunique()} stocks")

    for fold in folds():
        train = df[(df["trade_date"] >= fold.train_start) & (df["trade_date"] <= fold.train_end)]
        test  = df[(df["trade_date"] >= fold.test_start) & (df["trade_date"] <= fold.test_end)]
        if test.empty:
            continue

        preds, labels, pred_rows = [], [], []
        for symbol, sub in tqdm(test.groupby("symbol"), desc=f"rung0 {fold.fold_id}", leave=False):
            train_series = train[train["symbol"] == symbol]["log_ret_1d"]
            test_sub = sub
            if len(test_sub) > max_test_per_stock_per_fold:
                # Sample evenly across the test window
                step = max(1, len(test_sub) // max_test_per_stock_per_fold)
                test_sub = test_sub.iloc[::step].head(max_test_per_stock_per_fold)
            for _, row in test_sub.iterrows():
                hist = train_series  # ARIMA fit only on train (no leakage)
                p, c = predict_5d_direction(hist)
                preds.append(p)
                labels.append(int(row["y_5d_up"]))
                pred_rows.append((str(row["trade_date"]), row["company_key"], 0, fold.fold_id, float(p), int(c)))

        if not labels:
            continue
        with mlflow_run(rung=0, fold_id=fold.fold_id, model_family="arima", n_features=1) as ml:
            try:
                auc = float(roc_auc_score(labels, preds))
            except Exception:
                auc = 0.5
            acc = float(accuracy_score(labels, [int(p > 0.5) for p in preds]))
            ml.log_metric("test_auc", auc)
            ml.log_metric("accuracy", acc)
            ml.log_metric("n_test_rows", len(labels))
            print(f"  Rung 0 fold {fold.fold_id}: AUC={auc:.4f} acc={acc:.4f} n={len(labels)}")
        write_predictions(pred_rows)

if __name__ == "__main__":
    main()
