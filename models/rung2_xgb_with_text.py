#Rung 2: XGBoost on structured + text + sector-cross features.

from __future__ import annotations

import json
import warnings

import pandas as pd
import shap
import xgboost as xgb
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, roc_auc_score

from ingest.common import pg_conn
from models.mlflow_helpers import run as mlflow_run
from models.rung1_xgb_structured import STRUCTURED_FEATURES
from models.walkforward import folds

warnings.filterwarnings("ignore")

#  Feature sets 
LEGACY_SENTIMENT = [
    "finbert_news_mean_3d", "finbert_news_mean_30d", "finbert_press_30d",
    "n_news_3d", "n_8k_30d",
]
NEW_SENTIMENT = [
    "finbert_news_pos_3d", "finbert_news_neg_3d",
    "finbert_news_pos_30d", "finbert_news_neg_30d",
    "news_mean_change_5d", "news_disp_3d",
]
PCA_FEATURES = ["filing_pc1", "filing_pc2", "filing_pc3", "filing_pc4", "filing_pc5"]
SECTOR_DUMMIES = ["is_tech", "is_financials", "is_healthcare", "is_industrials", "is_consumer"]
CROSS_FEATURES = SECTOR_DUMMIES + ["senti_x_vol", "senti_x_absret", "newsvol_x_vol"]

FULL_FEATURES = STRUCTURED_FEATURES + LEGACY_SENTIMENT + NEW_SENTIMENT + PCA_FEATURES + CROSS_FEATURES

# Hyperparameters
#
# These intentionally diverge from Rung 1 (depth=5, lr=0.05, no reg_alpha, reg_lambda=1.0,
# colsample=0.8, min_child_weight=5, gamma=0.1). Rung 2 trains on 24 features versus
# Rung 1's 15, and many of the extra columns (per-class FinBERT probabilities, PCA
# components, cross-features) carry low signal-to-noise on a 5-day horizon. Sharing
# Rung 1's hyperparameters would have handicapped Rung 2 by under-regularising it.
#
# The harder regularisation (L1 alpha, larger L2 lambda, deeper pruning, smaller trees,
# slower learning, more aggressive column subsampling) is the standard prescription from
# Chen and Guestrin (2016) for tree ensembles on wider, noisier feature sets.
HP_RUNG2 = dict(
    n_estimators=300,
    max_depth=4,                # shallower trees: fewer noisy splits
    learning_rate=0.04,         # slower learning rate
    subsample=0.8,
    colsample_bytree=0.7,       # subsample columns more aggressively
    min_child_weight=8,         # require more support per leaf
    gamma=0.2,                  # higher pruning threshold
    reg_alpha=0.5,              # L1 to zero out dead text features
    reg_lambda=1.5,             # slightly stronger L2
    random_state=7,
    eval_metric="auc",
    enable_categorical=False,
    tree_method="hist",
    n_jobs=4,
)


def load_panel_full() -> pd.DataFrame:
    cols = ["trade_date", "symbol", "company_key", "y_5d_up"] + FULL_FEATURES
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


def train_fold(df: pd.DataFrame, fold, features: list[str]) -> tuple[float, float, list[tuple]]:
    # Train Rung 2 on a single fold; return (auc, acc, prediction rows).
    train = df[(df["trade_date"] >= fold.train_start) & (df["trade_date"] <= fold.train_end)]
    test = df[(df["trade_date"] >= fold.test_start) & (df["trade_date"] <= fold.test_end)]
    if test.empty or train.empty:
        return float("nan"), float("nan"), []

    X_train = train[features].fillna(0)
    y_train = train["y_5d_up"]
    X_test = test[features].fillna(0)
    y_test = test["y_5d_up"]

    clf = xgb.XGBClassifier(**HP_RUNG2)
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    probs = clf.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, probs))
    acc = float(accuracy_score(y_test, (probs > 0.5).astype(int)))

    # SHAP attribution on a 200-row sample.
    sample = X_test.head(min(200, len(X_test)))
    try:
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(sample)
    except Exception:
        sv = None

    with mlflow_run(rung=2, fold_id=fold.fold_id, model_family="xgboost",
                    n_features=len(features)) as ml:
        ml.log_metric("test_auc", auc)
        ml.log_metric("accuracy", acc)
        ml.log_metric("n_test_rows", len(y_test))
        for f, v in zip(features, clf.feature_importances_):
            ml.log_metric(f"fi_{f}", float(v))

    rows = []
    for i, (_, r) in enumerate(test.iterrows()):
        shap_payload = None
        if sv is not None and i < len(sample):
            row_sv = dict(zip(features, [float(x) for x in sv[i]]))
            top10 = dict(sorted(row_sv.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10])
            shap_payload = json.dumps(top10)
        rows.append((
            str(r["trade_date"]), r["company_key"], 2, fold.fold_id,
            float(probs[i]), int(probs[i] > 0.5), shap_payload,
        ))
    print(f"  Rung 2 fold {fold.fold_id}: AUC={auc:.4f} acc={acc:.4f} n={len(y_test)}")
    return auc, acc, rows


def persist_predictions(rows: list[tuple]) -> None:
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
        cur.executemany(
            "INSERT INTO gold.fct_predictions (date_key, company_key, model_rung, fold_id, prob_up, predicted_class, shap_json) "
            "VALUES (md5(%s), %s, %s, %s, %s, %s, %s)",
            rows,
        )


def paired_wilcoxon_vs_rung1(rung2_aucs: dict[str, float]) -> None:
    # Compare per-fold AUC of Rung 2 against the latest Rung 1 predictions in Postgres.
    if not rung2_aucs:
        return
    with pg_conn() as conn:
        rung1 = pd.read_sql(
            """
            SELECT p.fold_id, p.prob_up::float AS prob_up, fp.y_5d_up
            FROM gold.fct_predictions p
            JOIN gold.fct_feature_panel_daily fp
                 ON fp.company_key = p.company_key AND fp.date_key = p.date_key
            WHERE p.model_rung = 1
            """,
            conn,
        )
    if rung1.empty:
        print("[wilcoxon] no Rung 1 predictions in gold.fct_predictions yet; skipping")
        return
    rung1_aucs = {
        fold_id: float(roc_auc_score(g["y_5d_up"], g["prob_up"]))
        for fold_id, g in rung1.groupby("fold_id")
        if g["y_5d_up"].nunique() == 2
    }
    common = sorted(set(rung1_aucs) & set(rung2_aucs))
    if len(common) < 3:
        print(f"[wilcoxon] only {len(common)} matching folds, not enough for the test")
        return
    diffs = [rung2_aucs[f] - rung1_aucs[f] for f in common]
    stat, p = wilcoxon(diffs)
    mean_diff = sum(diffs) / len(diffs)
    print(f"\n[wilcoxon] paired test on {len(common)} folds: mean AUC delta (R2 - R1) = {mean_diff:+.4f}, "
          f"signed-rank statistic = {stat:.2f}, p = {p:.4f}")


def main() -> None:
    df = load_panel_full()
    print(f"Loaded {len(df)} panel rows, {df['symbol'].nunique()} stocks, {len(FULL_FEATURES)} features")

    # Clear any prior Rung 2 predictions so the table reflects the new model only.
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM gold.fct_predictions WHERE model_rung = 2")

    fold_aucs: dict[str, float] = {}
    all_rows: list[tuple] = []
    for fold in folds():
        auc, _, rows = train_fold(df, fold, FULL_FEATURES)
        if not pd.isna(auc):
            fold_aucs[fold.fold_id] = auc
        all_rows.extend(rows)

    persist_predictions(all_rows)
    print(f"Persisted {len(all_rows)} Rung 2 predictions over {len(fold_aucs)} folds")
    paired_wilcoxon_vs_rung1(fold_aucs)


if __name__ == "__main__":
    main()
