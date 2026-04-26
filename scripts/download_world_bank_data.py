"""Download public World Bank macro indicators for the dashboard."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DEFAULT_COUNTRIES = ["CHN", "USA", "GBR", "DEU", "JPN", "IND", "VNM"]
DEFAULT_START_YEAR = 2014
DEFAULT_END_YEAR = 2024

INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "gdp_growth_pct",
    "FP.CPI.TOTL.ZG": "inflation_pct",
    "SL.UEM.TOTL.ZS": "unemployment_pct",
    "NE.TRD.GNFS.ZS": "trade_pct_gdp",
    "BX.KLT.DINV.WD.GD.ZS": "fdi_inflows_pct_gdp",
}

WORLD_BANK_ENDPOINT = "https://api.worldbank.org/v2/country/{countries}/indicator/{indicator}"


def build_session() -> requests.Session:
    """Create a requests session with retries for transient API/network errors."""

    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    return session


def fetch_indicator(
    countries: Iterable[str],
    indicator: str,
    start_year: int,
    end_year: int,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch one World Bank indicator and return a tidy DataFrame."""

    session = session or build_session()
    countries_text = ";".join(countries)
    url = WORLD_BANK_ENDPOINT.format(countries=countries_text, indicator=indicator)
    params = {
        "format": "json",
        "date": f"{start_year}:{end_year}",
        "per_page": 20000,
    }
    response = session.get(url, params=params, timeout=45)
    response.raise_for_status()
    payload = response.json()

    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError(f"Unexpected World Bank API response for {indicator}: {payload!r}")

    records = payload[1] or []
    rows = []
    for item in records:
        rows.append(
            {
                "country": item["country"]["value"],
                "country_code": item["countryiso3code"],
                "year": int(item["date"]),
                "indicator_code": indicator,
                "indicator_name": item["indicator"]["value"],
                "metric": INDICATORS[indicator],
                "value": item["value"],
            }
        )
    return pd.DataFrame(rows)


def download_world_bank_data(
    root: Path,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    countries: Iterable[str] = DEFAULT_COUNTRIES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download all configured World Bank indicators and write raw files."""

    root = Path(root)
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    session = build_session()
    frames = [
        fetch_indicator(countries, indicator, start_year, end_year, session=session)
        for indicator in INDICATORS
    ]
    raw = pd.concat(frames, ignore_index=True)
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    raw = raw.sort_values(["country_code", "indicator_code", "year"]).reset_index(drop=True)

    accessed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metadata = pd.DataFrame(
        [
            {
                "accessed_at_utc": accessed_at,
                "source": "World Bank Open Data / World Development Indicators API",
                "endpoint": WORLD_BANK_ENDPOINT,
                "countries": ";".join(countries),
                "start_year": start_year,
                "end_year": end_year,
                "indicator_code": code,
                "metric": metric,
            }
            for code, metric in INDICATORS.items()
        ]
    )

    raw.to_csv(raw_dir / "world_bank_macro_raw.csv", index=False)
    metadata.to_csv(raw_dir / "world_bank_macro_metadata.csv", index=False)
    return raw, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument(
        "--countries",
        default=",".join(DEFAULT_COUNTRIES),
        help="Comma-separated ISO3 country codes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    countries = [country.strip().upper() for country in args.countries.split(",") if country.strip()]
    raw, metadata = download_world_bank_data(
        root=root,
        start_year=args.start_year,
        end_year=args.end_year,
        countries=countries,
    )
    print(f"Downloaded {len(raw):,} rows from World Bank API.")
    print(f"Raw data saved to {root / 'data' / 'raw' / 'world_bank_macro_raw.csv'}")
    print(f"Metadata saved to {root / 'data' / 'raw' / 'world_bank_macro_metadata.csv'}")
    print(f"Access timestamp: {metadata['accessed_at_utc'].iloc[0]}")


if __name__ == "__main__":
    main()
