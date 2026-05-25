"""Run every ingest module in sequence."""
from ingest import fetch_prices, fetch_fundamentals, fetch_news, fetch_press, fetch_sec


def main() -> None:
    print("=== prices ===")
    fetch_prices.main()
    print("=== fundamentals ===")
    fetch_fundamentals.main()
    print("=== news ===")
    fetch_news.main()
    print("=== press ===")
    fetch_press.main()
    print("=== sec ===")
    fetch_sec.main()
    print("=== done ===")


if __name__ == "__main__":
    main()
