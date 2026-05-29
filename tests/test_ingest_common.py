# Smoke tests for ingest.
from pathlib import Path

from ingest.common import all_tickers, load_universe, sector_for, sha256_of_file

def test_universe_has_20_tickers():
    assert len(all_tickers()) == 20

def test_universe_has_5_sectors():
    sectors = load_universe()
    assert len(sectors) == 5
    assert set(sectors.keys()) == {"Technology", "Financials", "Healthcare", "Industrials", "Consumer"}

def test_each_sector_has_four_stocks():
    sectors = load_universe()
    for name, tickers in sectors.items():
        assert len(tickers) == 4, f"sector {name} has {len(tickers)} tickers"

def test_sector_for_known_ticker():
    assert sector_for("AAPL") == "Technology"
    assert sector_for("JPM") == "Financials"
    assert sector_for("UNKNOWN_TICKER") == "Unknown"

def test_sha256_deterministic(tmp_path: Path):
    f = tmp_path / "x.txt"
    f.write_text("hello")
    assert sha256_of_file(f) == sha256_of_file(f)
    assert len(sha256_of_file(f)) == 64
