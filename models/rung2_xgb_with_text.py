"""Rung 2: XGBoost on ALL 30 features (structured + 5 sentiment + 5 PCA embedding).

Identical hyperparameters to Rung 1. The only difference is the feature set. Lift = text contribution."""
from __future__ import annotations

import warnings

from models.rung1_xgb_structured import STRUCTURED_FEATURES, HP, load_panel as _load_struct, train_and_score
from models.walkforward import folds
from ingest.common import pg_conn
import pandas as pd

warnings.filterwarnings("ignore")

TEXT_FEATURES = [
    # 5 sentiment
    "finbert_news_mean_3d", "finbert_news_mean_30d", "finbert_press_30d", "n_news_3d", "n_8k_30d",
    # 5 PCA embedding
    "filing_pc1", "filing_pc2", "filing_pc3", "filing_pc4", "filing_pc5",
]
FULL_FEATURES = STRUCTURED_FEATURES + TEXT_FEATURES

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

def main() -> None:
    df = load_panel_full()
    print(f"Loaded {len(df)} panel rows, {df['symbol'].nunique()} stocks, {len(FULL_FEATURES)} features")
    for fold in folds():
        train_and_score(df, fold, rung=2, features=FULL_FEATURES)

if __name__ == "__main__":
    main()
