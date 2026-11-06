"""Build processed ZCZ price tables."""

from cornsat.prices import expiry_dates, save_processed


def main() -> None:
    prices, front = save_processed()
    print(prices.groupby("contract").size().to_string())
    print("expiry:\n" + expiry_dates(prices).to_string())
    print(f"long: {len(prices)} rows; front: {len(front)} rows")
    print(f"front range: {front['date'].min().date()} .. {front['date'].max().date()}")


if __name__ == "__main__":
    main()
