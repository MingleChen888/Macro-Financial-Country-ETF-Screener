"""Create a macro-financial country ETF screening product."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analyze_macro_risk import add_scores, build_panel


COUNTRY_ORDER = ["USA", "CHN", "GBR", "DEU", "JPN", "IND", "VNM"]

METRIC_LABELS = {
    "macro_attractiveness_score": "Macro score",
    "ann_return_3y": "3Y annualized return",
    "ann_volatility_3y": "3Y annualized volatility",
    "sharpe_3y": "3Y Sharpe ratio",
    "max_drawdown_3y": "3Y max drawdown",
    "momentum_12m": "12M momentum",
    "macro_financial_score": "Macro-financial score",
}


def percentile_rank(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = series if higher_is_better else -series
    return values.rank(pct=True, method="average")


def load_macro_panel(root: Path) -> pd.DataFrame:
    raw_path = root / "data" / "raw" / "world_bank_macro_raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing {raw_path}. Run `python scripts/download_world_bank_data.py` first."
        )
    raw = pd.read_csv(raw_path)
    return add_scores(build_panel(raw))


def prepare_etf_prices(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ETF price history and a daily risk-free proxy from Yahoo data."""

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    etfs = prices.loc[prices["asset_type"].eq("country_etf")].copy()
    etfs = etfs.dropna(subset=["adj_close"]).sort_values(["ticker", "date"])
    etfs["daily_return"] = etfs.groupby("ticker")["adj_close"].pct_change()

    risk_free = prices.loc[prices["ticker"].eq("^IRX"), ["date", "adj_close"]].copy()
    risk_free = risk_free.rename(columns={"adj_close": "risk_free_yield_pct"})
    risk_free["risk_free_daily"] = risk_free["risk_free_yield_pct"].ffill() / 100 / 252
    return etfs, risk_free


def trailing_market_metrics(etfs: pd.DataFrame, risk_free: pd.DataFrame, years: int = 3) -> pd.DataFrame:
    """Compute return, risk, drawdown, and momentum metrics for each ETF."""

    latest_date = etfs["date"].max()
    start_date = latest_date - pd.DateOffset(years=years)
    trailing = etfs.loc[etfs["date"].ge(start_date)].merge(risk_free, on="date", how="left")
    trailing["risk_free_daily"] = trailing["risk_free_daily"].ffill().fillna(0)
    trailing["excess_return"] = trailing["daily_return"] - trailing["risk_free_daily"]

    rows = []
    for ticker, group in trailing.groupby("ticker"):
        group = group.sort_values("date").dropna(subset=["daily_return"])
        if len(group) < 120:
            continue
        first = group.iloc[0]
        last = group.iloc[-1]
        n_days = len(group)
        ann_return = (last["adj_close"] / first["adj_close"]) ** (252 / n_days) - 1
        ann_vol = group["daily_return"].std() * np.sqrt(252)
        excess_ann = group["excess_return"].mean() * 252
        sharpe = excess_ann / ann_vol if ann_vol and not np.isnan(ann_vol) else np.nan
        wealth = (1 + group["daily_return"]).cumprod()
        drawdown = wealth / wealth.cummax() - 1
        one_year_start = latest_date - pd.DateOffset(years=1)
        one_year = group.loc[group["date"].ge(one_year_start)]
        momentum_12m = (
            one_year["adj_close"].iloc[-1] / one_year["adj_close"].iloc[0] - 1
            if len(one_year) >= 60
            else np.nan
        )
        rows.append(
            {
                "country_code": first["country_code"],
                "country": first["country"],
                "ticker": ticker,
                "asset_name": first["asset_name"],
                "latest_market_date": latest_date.date().isoformat(),
                "market_window_start": group["date"].min().date().isoformat(),
                "market_window_days": n_days,
                "ann_return_3y": ann_return,
                "ann_volatility_3y": ann_vol,
                "sharpe_3y": sharpe,
                "max_drawdown_3y": drawdown.min(),
                "momentum_12m": momentum_12m,
            }
        )
    return pd.DataFrame(rows)


def monthly_return_panel(etfs: pd.DataFrame) -> pd.DataFrame:
    """Create a monthly return matrix for correlation analysis."""

    monthly_prices = (
        etfs.set_index("date")
        .groupby(["country_code", "ticker"])["adj_close"]
        .resample("ME")
        .last()
        .reset_index()
    )
    monthly_prices["label"] = monthly_prices["country_code"] + " (" + monthly_prices["ticker"] + ")"
    monthly = monthly_prices.pivot(index="date", columns="label", values="adj_close").pct_change()
    return monthly.dropna(how="all")


def latest_macro_panel(macro_panel: pd.DataFrame) -> pd.DataFrame:
    valid_counts = (
        macro_panel.dropna(subset=["macro_attractiveness_score"]).groupby("year").size()
    )
    latest_macro_year = int(valid_counts[valid_counts >= 4].index.max())
    cols = [
        "country",
        "country_code",
        "year",
        "macro_attractiveness_score",
        "gdp_growth_pct",
        "inflation_pct",
        "inflation_volatility_3y",
        "unemployment_pct",
        "trade_pct_gdp",
        "fdi_inflows_pct_gdp",
    ]
    return macro_panel.loc[macro_panel["year"].eq(latest_macro_year), cols].copy()


def build_macro_financial_scorecard(macro_latest: pd.DataFrame, market_metrics: pd.DataFrame) -> pd.DataFrame:
    scorecard = macro_latest.merge(market_metrics, on=["country_code", "country"], how="inner")
    rank_specs = {
        "macro_rank": ("macro_attractiveness_score", True),
        "return_rank": ("ann_return_3y", True),
        "sharpe_rank": ("sharpe_3y", True),
        "low_volatility_rank": ("ann_volatility_3y", False),
        "low_drawdown_rank": ("max_drawdown_3y", True),
        "momentum_rank": ("momentum_12m", True),
    }
    for rank_col, (metric, higher_is_better) in rank_specs.items():
        scorecard[rank_col] = percentile_rank(scorecard[metric], higher_is_better=higher_is_better)

    weights = {
        "macro_rank": 0.35,
        "return_rank": 0.20,
        "sharpe_rank": 0.15,
        "low_volatility_rank": 0.10,
        "low_drawdown_rank": 0.10,
        "momentum_rank": 0.10,
    }
    weighted = pd.Series(0.0, index=scorecard.index)
    available = pd.Series(0.0, index=scorecard.index)
    for col, weight in weights.items():
        valid = scorecard[col].notna()
        weighted = weighted.add(scorecard[col].fillna(0) * weight)
        available = available.add(np.where(valid, weight, 0.0))
    scorecard["macro_financial_score"] = np.where(available > 0, weighted / available * 100, np.nan)
    return scorecard.sort_values("macro_financial_score", ascending=False).reset_index(drop=True)


def save_score_chart(scorecard: pd.DataFrame, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid", font_scale=0.95)
    chart = scorecard.sort_values("macro_financial_score", ascending=True)
    plt.figure(figsize=(10.5, 6))
    plt.barh(chart["country"], chart["macro_financial_score"], color=sns.color_palette("mako", len(chart)))
    plt.title("Macro-financial attractiveness score")
    plt.xlabel("Score from 0 to 100 within selected country ETF universe")
    plt.ylabel("")
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(output_dir / "macro_financial_score_latest.png", dpi=180)
    plt.close()


def save_risk_return_chart(scorecard: pd.DataFrame, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid", font_scale=0.95)
    plt.figure(figsize=(9.5, 6.5))
    ax = sns.scatterplot(
        data=scorecard,
        x="ann_volatility_3y",
        y="ann_return_3y",
        size="macro_attractiveness_score",
        hue="country",
        sizes=(100, 500),
        alpha=0.85,
    )
    for _, row in scorecard.iterrows():
        ax.text(row["ann_volatility_3y"] + 0.003, row["ann_return_3y"], row["ticker"], fontsize=9)
    plt.title("Country ETF risk-return map with macro score as bubble size")
    plt.xlabel("3-year annualized volatility")
    plt.ylabel("3-year annualized return")
    plt.gca().xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    plt.gca().yaxis.set_major_formatter(lambda y, pos: f"{y:.0%}")
    plt.legend(title="Country", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "risk_return_macro_bubble.png", dpi=180)
    plt.close()


def save_drawdown_chart(etfs: pd.DataFrame, output_dir: Path) -> None:
    latest_date = etfs["date"].max()
    trailing = etfs.loc[etfs["date"].ge(latest_date - pd.DateOffset(years=3))].copy()
    trailing = trailing.dropna(subset=["daily_return"])
    trailing["wealth"] = trailing.groupby("ticker")["daily_return"].transform(lambda s: (1 + s).cumprod())
    trailing["drawdown"] = trailing.groupby("ticker")["wealth"].transform(lambda s: s / s.cummax() - 1)
    trailing["label"] = trailing["country_code"] + " (" + trailing["ticker"] + ")"

    sns.set_theme(style="whitegrid", font_scale=0.9)
    plt.figure(figsize=(11, 6))
    sns.lineplot(data=trailing, x="date", y="drawdown", hue="label")
    plt.title("Country ETF drawdowns over the latest 3-year window")
    plt.xlabel("Date")
    plt.ylabel("Drawdown from previous peak")
    plt.gca().yaxis.set_major_formatter(lambda y, pos: f"{y:.0%}")
    plt.legend(title="ETF", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "country_etf_drawdowns.png", dpi=180)
    plt.close()


def save_correlation_heatmap(monthly_returns: pd.DataFrame, output_dir: Path) -> None:
    corr = monthly_returns.corr()
    sns.set_theme(style="white", font_scale=0.9)
    plt.figure(figsize=(8.5, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Monthly return correlation across country ETFs")
    plt.tight_layout()
    plt.savefig(output_dir / "country_etf_return_correlation.png", dpi=180)
    plt.close()


def save_macro_financial_heatmap(scorecard: pd.DataFrame, output_dir: Path) -> None:
    heatmap_cols = [
        "macro_attractiveness_score",
        "ann_return_3y",
        "ann_volatility_3y",
        "sharpe_3y",
        "max_drawdown_3y",
        "momentum_12m",
    ]
    heatmap = scorecard.set_index("country")[heatmap_cols].rename(columns=METRIC_LABELS)
    sns.set_theme(style="white", font_scale=0.9)
    plt.figure(figsize=(11, 5.8))
    sns.heatmap(heatmap, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5)
    plt.title("Macro and market metrics behind the score")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_dir / "macro_financial_metric_heatmap.png", dpi=180)
    plt.close()


def write_product_brief(scorecard: pd.DataFrame, monthly_returns: pd.DataFrame, output_dir: Path) -> None:
    top = scorecard.iloc[0]
    bottom = scorecard.iloc[-1]
    corr = monthly_returns.corr()
    corr_pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
    strongest_pair = corr_pairs.sort_values(ascending=False).head(1)
    weakest_pair = corr_pairs.sort_values().head(1)
    strongest_text = (
        f"{strongest_pair.index[0][0]} and {strongest_pair.index[0][1]} "
        f"({strongest_pair.iloc[0]:.2f})"
        if not strongest_pair.empty
        else "not available"
    )
    weakest_text = (
        f"{weakest_pair.index[0][0]} and {weakest_pair.index[0][1]} "
        f"({weakest_pair.iloc[0]:.2f})"
        if not weakest_pair.empty
        else "not available"
    )

    display_cols = [
        "country",
        "ticker",
        "year",
        "latest_market_date",
        "macro_financial_score",
        "macro_attractiveness_score",
        "ann_return_3y",
        "ann_volatility_3y",
        "sharpe_3y",
        "max_drawdown_3y",
        "momentum_12m",
    ]
    brief = [
        "# Product Brief: Macro-Financial Country ETF Screener",
        "",
        "## User-facing purpose",
        "",
        (
            "This product helps a junior finance or strategy analyst compare selected "
            "countries using both economic fundamentals and investable ETF market signals."
        ),
        "",
        "## Main takeaway",
        "",
        (
            f"{top['country']} ({top['ticker']}) has the highest combined macro-financial "
            f"score at {top['macro_financial_score']:.1f} out of 100. "
            f"{bottom['country']} ({bottom['ticker']}) has the lowest score at "
            f"{bottom['macro_financial_score']:.1f}. The top result is driven by the "
            "combination of macro fundamentals and ETF risk-return characteristics, not by "
            "one single indicator."
        ),
        "",
        "## Diversification note",
        "",
        (
            f"The strongest monthly ETF return correlation is {strongest_text}. The weakest "
            f"pairwise correlation is {weakest_text}. Lower correlations can be useful for "
            "diversification, but this should be checked with longer samples and additional "
            "risk factors."
        ),
        "",
        "## Latest scorecard",
        "",
        scorecard[display_cols].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Important caution",
        "",
        (
            "This is an educational screening tool, not financial advice. ETF returns can "
            "diverge from the underlying domestic economy because of currency exposure, "
            "index composition, valuation changes, and global investor sentiment."
        ),
        "",
    ]
    (output_dir / "macro_financial_product_brief.md").write_text("\n".join(brief), encoding="utf-8")


def run_analysis(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(root)
    raw_market_path = root / "data" / "raw" / "yahoo_country_etf_prices_raw.csv"
    processed_dir = root / "data" / "processed"
    output_dir = root / "outputs"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_market_path.exists():
        raise FileNotFoundError(
            f"Missing {raw_market_path}. Run `python scripts/download_yahoo_finance_data.py` first."
        )

    macro_panel = load_macro_panel(root)
    prices = pd.read_csv(raw_market_path)
    etfs, risk_free = prepare_etf_prices(prices)
    market_metrics = trailing_market_metrics(etfs, risk_free, years=3)
    monthly_returns = monthly_return_panel(etfs)
    macro_latest = latest_macro_panel(macro_panel)
    scorecard = build_macro_financial_scorecard(macro_latest, market_metrics)

    macro_panel.to_csv(processed_dir / "macro_business_panel.csv", index=False)
    market_metrics.to_csv(processed_dir / "country_etf_market_metrics.csv", index=False)
    monthly_returns.to_csv(processed_dir / "country_etf_monthly_returns.csv")
    scorecard.to_csv(output_dir / "latest_macro_financial_scorecard.csv", index=False)

    save_score_chart(scorecard, output_dir)
    save_risk_return_chart(scorecard, output_dir)
    save_drawdown_chart(etfs, output_dir)
    save_correlation_heatmap(monthly_returns, output_dir)
    save_macro_financial_heatmap(scorecard, output_dir)
    write_product_brief(scorecard, monthly_returns, output_dir)
    return scorecard, market_metrics, monthly_returns


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    scorecard, market_metrics, monthly_returns = run_analysis(root)
    print(f"Country ETF market rows: {len(market_metrics):,}")
    print(f"Monthly return observations: {len(monthly_returns):,}")
    print("Top macro-financial scorecard:")
    print(
        scorecard[
            ["country", "ticker", "macro_financial_score", "ann_return_3y", "ann_volatility_3y", "sharpe_3y"]
        ]
        .head(5)
        .to_string(index=False)
    )
    print(f"Outputs saved to {root / 'outputs'}")


if __name__ == "__main__":
    main()
