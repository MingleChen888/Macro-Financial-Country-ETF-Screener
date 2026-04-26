"""Build the macro business risk dashboard outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


METRIC_LABELS = {
    "gdp_growth_pct": "GDP growth (%)",
    "inflation_pct": "Inflation (%)",
    "unemployment_pct": "Unemployment (%)",
    "trade_pct_gdp": "Trade (% of GDP)",
    "fdi_inflows_pct_gdp": "FDI inflows (% of GDP)",
    "inflation_volatility_3y": "3-year inflation volatility",
    "macro_attractiveness_score": "Macro attractiveness score",
}


def percentile_rank(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Return percentile ranks where higher output always means more attractive."""

    values = series if higher_is_better else -series
    return values.rank(pct=True, method="average")


def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean World Bank data and build a country-year panel."""

    panel = (
        raw.pivot_table(
            index=["country_code", "country", "year"],
            columns="metric",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .sort_values(["country_code", "year"])
    )
    panel.columns.name = None

    numeric_cols = [
        "gdp_growth_pct",
        "inflation_pct",
        "unemployment_pct",
        "trade_pct_gdp",
        "fdi_inflows_pct_gdp",
    ]
    for col in numeric_cols:
        if col in panel:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")

    panel["inflation_volatility_3y"] = (
        panel.groupby("country_code")["inflation_pct"]
        .rolling(window=3, min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
    )
    return panel


def add_scores(panel: pd.DataFrame) -> pd.DataFrame:
    """Add normalized score components and the final macro score."""

    scored = panel.copy()
    score_specs = {
        "growth_rank": ("gdp_growth_pct", True),
        "trade_rank": ("trade_pct_gdp", True),
        "fdi_rank": ("fdi_inflows_pct_gdp", True),
        "low_unemployment_rank": ("unemployment_pct", False),
        "low_inflation_volatility_rank": ("inflation_volatility_3y", False),
    }
    for rank_col, (metric_col, higher_is_better) in score_specs.items():
        scored[rank_col] = scored.groupby("year")[metric_col].transform(
            lambda s: percentile_rank(s, higher_is_better=higher_is_better)
        )

    weights = {
        "growth_rank": 0.30,
        "trade_rank": 0.20,
        "fdi_rank": 0.15,
        "low_unemployment_rank": 0.20,
        "low_inflation_volatility_rank": 0.15,
    }
    available_weight = pd.Series(0.0, index=scored.index)
    weighted_score = pd.Series(0.0, index=scored.index)
    for col, weight in weights.items():
        valid = scored[col].notna()
        available_weight = available_weight.add(np.where(valid, weight, 0.0))
        weighted_score = weighted_score.add(scored[col].fillna(0) * weight)

    scored["macro_attractiveness_score"] = np.where(
        available_weight > 0,
        (weighted_score / available_weight) * 100,
        np.nan,
    )
    return scored


def latest_scorecard(scored: pd.DataFrame) -> pd.DataFrame:
    """Return the latest year with at least four countries having scores."""

    valid_counts = scored.dropna(subset=["macro_attractiveness_score"]).groupby("year").size()
    eligible_years = valid_counts[valid_counts >= 4].index
    if len(eligible_years) == 0:
        raise ValueError("No year has enough valid observations for a scorecard.")
    latest_year = int(max(eligible_years))
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
    return (
        scored.loc[scored["year"].eq(latest_year), cols]
        .sort_values("macro_attractiveness_score", ascending=False)
        .reset_index(drop=True)
    )


def save_gdp_growth_chart(scored: pd.DataFrame, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid", font_scale=0.95)
    plt.figure(figsize=(11, 6))
    sns.lineplot(data=scored, x="year", y="gdp_growth_pct", hue="country", marker="o")
    plt.axhline(0, color="#333333", linewidth=0.8)
    plt.title("GDP growth trend across selected economies")
    plt.xlabel("Year")
    plt.ylabel("GDP growth, annual %")
    plt.legend(title="Country", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "gdp_growth_trends.png", dpi=180)
    plt.close()


def save_latest_score_chart(scorecard: pd.DataFrame, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid", font_scale=0.95)
    plt.figure(figsize=(10, 6))
    chart = scorecard.sort_values("macro_attractiveness_score", ascending=True)
    colors = sns.color_palette("viridis", n_colors=len(chart))
    plt.barh(chart["country"], chart["macro_attractiveness_score"], color=colors)
    plt.title(f"Macro attractiveness score, {int(scorecard['year'].iloc[0])}")
    plt.xlabel("Score from 0 to 100 among selected countries")
    plt.ylabel("")
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(output_dir / "macro_score_latest.png", dpi=180)
    plt.close()


def save_heatmap(scorecard: pd.DataFrame, output_dir: Path) -> None:
    heatmap_cols = [
        "gdp_growth_pct",
        "inflation_pct",
        "inflation_volatility_3y",
        "unemployment_pct",
        "trade_pct_gdp",
        "fdi_inflows_pct_gdp",
    ]
    heatmap_data = scorecard.set_index("country")[heatmap_cols]
    heatmap_data = heatmap_data.rename(columns=METRIC_LABELS)

    sns.set_theme(style="white", font_scale=0.9)
    plt.figure(figsize=(11, 5.5))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"label": "Raw indicator value"},
    )
    plt.title(f"Latest macro indicators, {int(scorecard['year'].iloc[0])}")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_dir / "macro_indicator_heatmap.png", dpi=180)
    plt.close()


def write_product_brief(scorecard: pd.DataFrame, output_dir: Path) -> None:
    latest_year = int(scorecard["year"].iloc[0])
    top = scorecard.iloc[0]
    bottom = scorecard.iloc[-1]
    lines = [
        "# Product Brief: Macro Business Risk Dashboard",
        "",
        f"Latest scorecard year: {latest_year}",
        "",
        "## Main takeaway",
        "",
        (
            f"Among the selected economies, {top['country']} has the strongest macro "
            f"screening score in {latest_year}, with a score of "
            f"{top['macro_attractiveness_score']:.1f} out of 100. "
            f"{bottom['country']} has the weakest score in this comparison, with "
            f"{bottom['macro_attractiveness_score']:.1f} out of 100."
        ),
        "",
        "## How to use this product",
        "",
        (
            "Use the scorecard as a first screening tool for business expansion research. "
            "A higher score means the country performed better among this peer group on "
            "growth, trade openness, FDI inflows, unemployment, and inflation stability."
        ),
        "",
        "## Important caution",
        "",
        (
            "This is not a final market-entry recommendation. The user should next review "
            "sector-level demand, regulation, exchange-rate risk, political risk, competitor "
            "conditions, and firm-specific strategic fit."
        ),
        "",
        "## Latest scorecard",
        "",
        scorecard.to_markdown(index=False, floatfmt=".2f"),
        "",
    ]
    (output_dir / "product_brief.md").write_text("\n".join(lines), encoding="utf-8")


def run_analysis(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full analysis pipeline from raw CSV to dashboard outputs."""

    root = Path(root)
    raw_path = root / "data" / "raw" / "world_bank_macro_raw.csv"
    processed_dir = root / "data" / "processed"
    output_dir = root / "outputs"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing {raw_path}. Run `python scripts/download_world_bank_data.py` first."
        )

    raw = pd.read_csv(raw_path)
    panel = add_scores(build_panel(raw))
    scorecard = latest_scorecard(panel)

    panel.to_csv(processed_dir / "macro_business_panel.csv", index=False)
    scorecard.to_csv(output_dir / "latest_macro_scorecard.csv", index=False)
    save_gdp_growth_chart(panel, output_dir)
    save_latest_score_chart(scorecard, output_dir)
    save_heatmap(scorecard, output_dir)
    write_product_brief(scorecard, output_dir)
    return panel, scorecard


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    panel, scorecard = run_analysis(root)
    print(f"Processed panel rows: {len(panel):,}")
    print(f"Latest scorecard year: {int(scorecard['year'].iloc[0])}")
    print("Top three countries:")
    print(scorecard[["country", "macro_attractiveness_score"]].head(3).to_string(index=False))
    print(f"Outputs saved to {root / 'outputs'}")


if __name__ == "__main__":
    main()

