# Macro-Financial Country ETF Screener

This repository is a Python-based data analysis project for **ACC102 Mini Assignment Track 2: GitHub Data Analysis Project**.

The project has been upgraded from a simple macro dashboard into a macro-financial screening product. It combines public World Bank macroeconomic indicators with Yahoo Finance country ETF market data to compare selected economies from both an economic and financial perspective.

## Product Idea

The product is designed for a junior finance or strategy analyst who needs a first-stage screen of country opportunities. It does not make investment recommendations. Instead, it helps the user compare selected countries using:

- macroeconomic fundamentals from World Bank Open Data,
- investable country ETF performance from Yahoo Finance,
- risk-return indicators such as annualized return, volatility, Sharpe ratio, drawdown, and momentum,
- return correlation across country ETFs for a simple diversification view.

## Analytical Question

Which selected countries look more attractive when we evaluate both macroeconomic fundamentals and financial market risk-return signals?

## Target User

The target user is a junior analyst preparing an initial country allocation or market-entry screening report. The output can help the user decide which countries deserve deeper research, but it should not be treated as financial advice.

## Data Sources

### World Bank

Data are collected from the World Bank Open Data API:

- Source: World Bank Open Data / World Development Indicators
- API endpoint pattern: `https://api.worldbank.org/v2/country/{countries}/indicator/{indicator}?format=json`
- Access date: generated at runtime and recorded in `data/raw/world_bank_macro_metadata.csv`

World Bank indicators:

| Code | Indicator |
| --- | --- |
| `NY.GDP.MKTP.KD.ZG` | GDP growth, annual % |
| `FP.CPI.TOTL.ZG` | Inflation, consumer prices, annual % |
| `SL.UEM.TOTL.ZS` | Unemployment, total % of total labor force |
| `NE.TRD.GNFS.ZS` | Trade % of GDP |
| `BX.KLT.DINV.WD.GD.ZS` | Foreign direct investment, net inflows % of GDP |

### Yahoo Finance

Daily market data are collected from the Yahoo Finance public chart API:

- API endpoint pattern: `https://query1.finance.yahoo.com/v8/finance/chart/{ticker}`
- Access date: generated at runtime and recorded in `data/raw/yahoo_country_etf_metadata.csv`

Country ETF mapping:

| Country | Ticker | ETF |
| --- | --- | --- |
| United States | `SPY` | SPDR S&P 500 ETF Trust |
| China | `MCHI` | iShares MSCI China ETF |
| United Kingdom | `EWU` | iShares MSCI United Kingdom ETF |
| Germany | `EWG` | iShares MSCI Germany ETF |
| Japan | `EWJ` | iShares MSCI Japan ETF |
| India | `INDA` | iShares MSCI India ETF |
| Viet Nam | `VNM` | VanEck Vietnam ETF |

The script also downloads `^IRX` as a rough 13-week Treasury Bill yield proxy for excess-return and Sharpe-ratio calculations.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── data
│   ├── processed
│   └── raw
├── notebooks
│   └── macro_financial_country_etf_workflow.ipynb
├── outputs
├── reports
│   ├── demo_video_script.md
│   ├── reflection_report_draft.md
│   └── submission_checklist.md
└── scripts
    ├── analyze_macro_finance.py
    ├── analyze_macro_risk.py
    ├── download_world_bank_data.py
    └── download_yahoo_finance_data.py
```

`analyze_macro_risk.py` is kept because the macro-financial analysis imports its macro cleaning and scoring helper functions.

## How To Run

Create or activate a Python environment, install packages, then run the full pipeline:

```bash
python -m pip install -r requirements.txt
python scripts/download_world_bank_data.py
python scripts/download_yahoo_finance_data.py
python scripts/analyze_macro_finance.py
```

Use the same Python interpreter for `python -m pip` and the three `python scripts/...` commands. If a Mac opens system Python without pandas, activate Anaconda or another environment first.

## Main Outputs

The project generates:

- `data/raw/world_bank_macro_raw.csv`
- `data/raw/world_bank_macro_metadata.csv`
- `data/raw/yahoo_country_etf_prices_raw.csv`
- `data/raw/yahoo_country_etf_metadata.csv`
- `data/processed/macro_business_panel.csv`
- `data/processed/country_etf_market_metrics.csv`
- `data/processed/country_etf_monthly_returns.csv`
- `outputs/latest_macro_financial_scorecard.csv`
- `outputs/macro_financial_score_latest.png`
- `outputs/risk_return_macro_bubble.png`
- `outputs/country_etf_drawdowns.png`
- `outputs/country_etf_return_correlation.png`
- `outputs/macro_financial_metric_heatmap.png`
- `outputs/macro_financial_product_brief.md`

You can also open and run `notebooks/macro_financial_country_etf_workflow.ipynb` as the notebook evidence for the Python workflow.

## Method Summary

1. Download macroeconomic indicators from the World Bank API.
2. Download daily country ETF prices and a Treasury bill yield proxy from Yahoo Finance.
3. Clean and reshape World Bank data into a country-year macro panel.
4. Create macro metrics including three-year inflation volatility and a macro attractiveness score.
5. Convert ETF prices into daily and monthly returns.
6. Compute three-year annualized return, annualized volatility, Sharpe ratio, maximum drawdown, and 12-month momentum.
7. Combine macro and market ranks into a macro-financial score.
8. Generate a scorecard, charts, a correlation heatmap, and a short product brief.

The combined score uses transparent weights:

```text
35% macro attractiveness rank
20% 3-year annualized ETF return rank
15% 3-year Sharpe ratio rank
10% low 3-year volatility rank
10% low 3-year maximum drawdown rank
10% 12-month momentum rank
```

Higher values indicate a stronger result within the selected country ETF universe.

## Current Result Snapshot

The latest generated scorecard uses World Bank macro data through 2024 and Yahoo Finance ETF data through the latest available trading day at runtime. In the latest run, Japan, the United States, and Viet Nam ranked highest by the combined macro-financial score.

## Limitations

This project is an educational screening tool, not financial advice. It uses a small country universe and simple transparent weights. ETF returns can diverge from domestic macroeconomic performance because of currency exposure, index composition, valuation changes, investor sentiment, and global risk conditions. World Bank data may also be revised or lagged. A real investment or market-entry decision would require deeper sector, regulatory, political, currency, and firm-level analysis.
