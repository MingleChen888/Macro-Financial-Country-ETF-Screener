"""Download public Yahoo Finance market data for country ETFs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


DEFAULT_START_DATE = "2019-01-01"

COUNTRY_ETFS = [
    {
        "country_code": "USA",
        "country": "United States",
        "ticker": "SPY",
        "asset_name": "SPDR S&P 500 ETF Trust",
    },
    {
        "country_code": "CHN",
        "country": "China",
        "ticker": "MCHI",
        "asset_name": "iShares MSCI China ETF",
    },
    {
        "country_code": "GBR",
        "country": "United Kingdom",
        "ticker": "EWU",
        "asset_name": "iShares MSCI United Kingdom ETF",
    },
    {
        "country_code": "DEU",
        "country": "Germany",
        "ticker": "EWG",
        "asset_name": "iShares MSCI Germany ETF",
    },
    {
        "country_code": "JPN",
        "country": "Japan",
        "ticker": "EWJ",
        "asset_name": "iShares MSCI Japan ETF",
    },
    {
        "country_code": "IND",
        "country": "India",
        "ticker": "INDA",
        "asset_name": "iShares MSCI India ETF",
    },
    {
        "country_code": "VNM",
        "country": "Viet Nam",
        "ticker": "VNM",
        "asset_name": "VanEck Vietnam ETF",
    },
]

RISK_FREE_PROXY = {
    "country_code": "USA",
    "country": "United States",
    "ticker": "^IRX",
    "asset_name": "13 Week Treasury Bill yield proxy",
}

YAHOO_CHART_ENDPOINT = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"


def to_unix_timestamp(date_text: str) -> int:
    dt = datetime.fromisoformat(date_text).replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def fetch_chart(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily price history from Yahoo Finance's public chart endpoint."""

    url = YAHOO_CHART_ENDPOINT.format(ticker=ticker)
    params = {
        "period1": to_unix_timestamp(start_date),
        "period2": to_unix_timestamp(end_date),
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    response = requests.get(
        url,
        params=params,
        timeout=45,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()
    payload = response.json()
    result = payload.get("chart", {}).get("result") or []
    if not result:
        error = payload.get("chart", {}).get("error")
        raise ValueError(f"No Yahoo Finance chart data for {ticker}. Error: {error}")

    chart = result[0]
    timestamps = chart.get("timestamp") or []
    quote = (chart.get("indicators", {}).get("quote") or [{}])[0]
    adjclose = (chart.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")
    if not timestamps:
        raise ValueError(f"Yahoo Finance returned no timestamps for {ticker}.")

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps, unit="s", utc=True).date,
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
            "adj_close": adjclose,
        }
    )
    frame["ticker"] = ticker
    return frame


def download_yahoo_finance_data(
    root: Path,
    start_date: str = DEFAULT_START_DATE,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download ETF price data and a Treasury bill yield proxy."""

    root = Path(root)
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    end_date = end_date or datetime.now(timezone.utc).date().isoformat()
    assets = COUNTRY_ETFS + [RISK_FREE_PROXY]
    frames = []
    for asset in assets:
        frame = fetch_chart(asset["ticker"], start_date=start_date, end_date=end_date)
        frame["country_code"] = asset["country_code"]
        frame["country"] = asset["country"]
        frame["asset_name"] = asset["asset_name"]
        frame["asset_type"] = (
            "risk_free_proxy" if asset["ticker"] == RISK_FREE_PROXY["ticker"] else "country_etf"
        )
        frames.append(frame)

    prices = pd.concat(frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"])
    numeric_cols = ["open", "high", "low", "close", "volume", "adj_close"]
    for col in numeric_cols:
        prices[col] = pd.to_numeric(prices[col], errors="coerce")
    prices = prices.sort_values(["asset_type", "ticker", "date"]).reset_index(drop=True)

    accessed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metadata = pd.DataFrame(
        [
            {
                "accessed_at_utc": accessed_at,
                "source": "Yahoo Finance public chart API",
                "endpoint": YAHOO_CHART_ENDPOINT,
                "ticker": asset["ticker"],
                "country_code": asset["country_code"],
                "country": asset["country"],
                "asset_name": asset["asset_name"],
                "start_date": start_date,
                "end_date": end_date,
            }
            for asset in assets
        ]
    )

    prices.to_csv(raw_dir / "yahoo_country_etf_prices_raw.csv", index=False)
    metadata.to_csv(raw_dir / "yahoo_country_etf_metadata.csv", index=False)
    return prices, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    prices, metadata = download_yahoo_finance_data(
        root=root,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    tickers = ", ".join(metadata["ticker"].tolist())
    print(f"Downloaded {len(prices):,} rows from Yahoo Finance for: {tickers}.")
    print(f"Raw data saved to {root / 'data' / 'raw' / 'yahoo_country_etf_prices_raw.csv'}")
    print(f"Metadata saved to {root / 'data' / 'raw' / 'yahoo_country_etf_metadata.csv'}")
    print(f"Access timestamp: {metadata['accessed_at_utc'].iloc[0]}")


if __name__ == "__main__":
    main()

