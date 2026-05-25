"""Rung 1: XGBoost on the 20 STRUCTURED features (10 price + 10 fundamentals).

Hyperparameters fixed; same values reused in Rung 2. Lift from Rung 1->2 measures text-feature contribution."""
from __future__ import annotations

import json
import warnings

import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

from ingest.common import pg_conn
from models.mlflow_helpers import run as mlflow_run
from models.walkforward import folds

warnings.filterwarnings("ignore")

STRUCTURED_FEATURES = [
    # 10 price features
    "log_ret_1d", "sma_5", "sma_20", "sma_50", "ema_12", "ema_26",
    "rsi_14", "macd_hist", "bb_z", "vol_20d",
    # 10 fundamentals features
    "roe", "roa", "debt_eq", "gross_margin", "op_margin", "asset_turnover",
    "pe_ttm_proxy", "pb_proxy", "ebitda_to_ev_proxy", "fcf_to_assets",
]

HP = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    reg_lambda=1.0,
    random_state=7,
    eval_metric="auc",
    enable_categorical=False,
    tree_method="hist",
    n_jobs=4,
)

def load_panel() -> pd.DataFrame:
    cols = ["trade_date", "symbol", "company_key", "y_5d_up"] + STRUCTURED_FEATURES
    with pg_conn() as conn:
        return pd.read_sql(
            f"""
            SELECT {", ".join(cols)}
            FROM gold.fct_feature_panel_daily
            WHERE y_5d_up IS NOT NULL
            ORDER BY symbol, trade_date
            """,
            conn,
        )

def train_and_score(df: pd.DataFrame, fold, rung: int, features: list[str]) -> None:
    train = df[(df["trade_date"] >= fold.train_start) & (df["trade_date"] <= fold.train_end)]
    test  = df[(df["trade_date"] >= fold.test_start) & (df["trade_date"] <= fold.test_end)]
    if test.empty or train.empty:
        return

    X_train = train[features].fillna(0)
    y_train = train["y_5d_up"]
    X_test = test[features].fillna(0)
    y_test = test["y_5d_up"]

    clf = xgb.XGBClassifier(**HP)
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    probs = clf.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, probs))
    acc = float(accuracy_score(y_test, (probs > 0.5).astype(int)))

    # SHAP top-10 per row (sampled for speed)
    sample = X_test.head(min(200, len(X_test)))
    try:
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(sample)
    except Exception:
        sv = None

    with mlflow_run(rung=rung, fold_id=fold.fold_id, model_family="xgboost", n_features=len(features)) as ml:
        ml.log_metric("test_auc", auc)
        ml.log_metric("accuracy", acc)
        ml.log_metric("n_test_rows", len(y_test))
        # Feature importance as a metric snapshot
        imps = clf.feature_importances_
        for f, v in zip(features, imps):
            ml.log_metric(f"fi_{f}", float(v))

    # Persist predictions
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
        rows = []
        for i, (_, r) in enumerate(test.iterrows()):
            shap_payload = None
            if sv is not None and i < len(sample):
                row_sv = dict(zip(features, [float(x) for x in sv[i]]))
                # Top 10 by abs
                top10 = dict(sorted(row_sv.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10])
                shap_payload = json.dumps(top10)
            rows.append((str(r["trade_date"]), r["company_key"], rung, fold.fold_id, float(probs[i]), int(probs[i] > 0.5), shap_payload))
        cur.executemany(
            "INSERT INTO gold.fct_predictions (date_key, company_key, model_rung, fold_id, prob_up, predicted_class, shap_json) VALUES (md5(%s), %s, %s, %s, %s, %s, %s)",
            rows,
        )
    print(f"  Rung {rung} fold {fold.fold_id}: AUC={auc:.4f} acc={acc:.4f} n={len(y_test)}")

def main() -> None:
    df = load_panel()
    print(f"Loaded {len(df)} panel rows, {df['symbol'].nunique()} stocks, {len(STRUCTURED_FEATURES)} features")
    for fold in folds():
        train_and_score(df, fold, rung=1, features=STRUCTURED_FEATURES)

if __name__ == "__main__":
    main()
