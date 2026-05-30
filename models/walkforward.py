# Walk-forward fold generator.
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

@dataclass
class Fold:
    fold_id: str
    train_start: date
    train_end: date
    test_start: date
    test_end: date

def folds(
    start: date = date(2021, 1, 1),
    end: date = date(2025, 12, 31),
    train_years: int = 3,
    test_quarter_days: int = 63,
    # 10 calendar days = embargo for the 5-trading-day target (~7 cal days)
    # with a 3-day safety buffer for holidays. Prevents the last train rows'
    # y_5d_up targets from touching the first 1-2 test trading days.
    gap_days: int = 10,
) -> list[Fold]:
    out: list[Fold] = []
    cursor = start + timedelta(days=365 * train_years)
    while cursor + timedelta(days=test_quarter_days) <= end:
        train_end = cursor - timedelta(days=gap_days)
        train_start = train_end - timedelta(days=365 * train_years)
        test_start = cursor
        test_end = cursor + timedelta(days=test_quarter_days)
        out.append(
            Fold(
                fold_id=test_start.isoformat(),
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        cursor = cursor + timedelta(days=test_quarter_days)
    return out

if __name__ == "__main__":
    for f in folds():
        print(
            f"{f.fold_id}: train {f.train_start} -> {f.train_end} | "
            f"test {f.test_start} -> {f.test_end}"
        )
