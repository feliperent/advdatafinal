# Leakage guard for gold.fct_feature_panel_daily.
#
# Every feature on every panel row must have been knowable on or before that row's
# trade_date. as_of_date is the GREATEST(...) of every source's as_of_date used to
# build the row; if it is ever later than trade_date, the row leaked information
# from the future. This test queries the panel and asserts the count of bad rows
# is zero. Cited by report Section 3.2 (Data governance > Leakage prevention).
from __future__ import annotations

import os

import pytest

from ingest.common import pg_conn


@pytest.mark.skipif(
    not os.getenv("DATABASE_URL") and not os.getenv("PG_PASSWORD"),
    reason="Postgres not configured (no DATABASE_URL or PG_PASSWORD in env)",
)
def test_no_feature_panel_row_uses_future_information() -> None:
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) FROM gold.fct_feature_panel_daily
            WHERE as_of_date > trade_date
            """
        )
        (bad_rows,) = cur.fetchone()
    assert bad_rows == 0, (
        f"{bad_rows} panel rows have as_of_date > trade_date. "
        "A feature in those rows was computed from data that did not yet exist "
        "on the row's trading day. Check the asof-join logic in "
        "dbt/models/gold/fct_feature_panel_daily.sql."
    )


@pytest.mark.skipif(
    not os.getenv("DATABASE_URL") and not os.getenv("PG_PASSWORD"),
    reason="Postgres not configured",
)
def test_walkforward_gap_embargoes_the_target() -> None:
    # The y_5d_up target uses LEAD(close_px, 5) over trade_date -- 5 TRADING days,
    # ~7 calendar days. The walk-forward gap must be at least that wide so the last
    # training row's target does not touch the first test trading day.
    from models.walkforward import folds

    # 5 trading days ~ 7 calendar days; require >= 7 to leave no overlap.
    MIN_GAP_DAYS = 7
    for fold in folds():
        gap = (fold.test_start - fold.train_end).days
        assert gap >= MIN_GAP_DAYS, (
            f"Fold {fold.fold_id} has gap={gap} calendar days between train_end and "
            f"test_start, but the 5-trading-day target needs at least {MIN_GAP_DAYS} "
            "to avoid leakage."
        )
