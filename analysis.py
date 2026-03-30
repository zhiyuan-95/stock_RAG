def _fetch_sector_snapshot(ticker):
    import ingest_stock

    ticker_map = ingest_stock._load_sec_ticker_map()
    sec_identity = ticker_map.get(ticker)
    if not sec_identity:
        return {
            "Ticker": ticker,
            "Sector": None,
            "SIC": None,
            "SIC Description": None,
        }

    cik = sec_identity["cik"]
    submissions = ingest_stock._sec_get_json(ingest_stock.SEC_SUBMISSIONS_URL_TEMPLATE.format(cik=cik))
    sic = ingest_stock._clean_profile_field(submissions.get("sic"))
    sic_description = ingest_stock._clean_profile_field(submissions.get("sicDescription"))
    return {
        "Ticker": ticker,
        "Sector": ingest_stock._derive_sector_from_sic(sic, sic_description),
        "SIC": sic,
        "SIC Description": sic_description,
    }


def _load_annual_financial_history_from_sql(ticker_filter=None):
    import sqlite3
    import pandas as pd
    import ingest_stock

    conn = sqlite3.connect(ingest_stock.DEFAULT_STOCK_DB_PATH)
    if ticker_filter:
        placeholders = ",".join("?" for _ in ticker_filter)
        query = f"""
        SELECT * FROM financial_indicators
        WHERE Frequency = 'Annual' AND Ticker IN ({placeholders})
        """
        df = pd.read_sql_query(query, conn, params=list(ticker_filter))
    else:
        query = """
        SELECT * FROM financial_indicators
        WHERE Frequency = 'Annual'
        """
        df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        return df

    df["Ticker"] = df["Ticker"].astype(str).str.upper()
    df["Period End Date"] = pd.to_datetime(df["Period End Date"], errors="coerce")
    df = df.dropna(subset=["Ticker", "Period End Date"]).copy()
    df["Fiscal Year"] = df["Period End Date"].dt.year
    return df


def _build_sql_sector_snapshot_frame(tickers, max_workers=8):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import pandas as pd

    snapshots = []
    tickers = sorted({str(ticker).upper() for ticker in tickers if ticker})

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_fetch_sector_snapshot, ticker): ticker for ticker in tickers}
        for future in as_completed(future_map):
            ticker = future_map[future]
            try:
                snapshot = future.result()
            except Exception:
                snapshot = {
                    "Ticker": ticker,
                    "Sector": None,
                    "SIC": None,
                    "SIC Description": None,
                }
            snapshots.append(snapshot)

    return pd.DataFrame(snapshots)


def _add_revenue_cagr_5yr(history_df):
    import pandas as pd

    if history_df.empty:
        history_df["Revenue CAGR 5-Year"] = pd.Series(dtype="float64")
        return history_df

    working = history_df.sort_values(["Ticker", "Period End Date"]).copy()
    working["Revenue CAGR 5-Year"] = pd.Series(index=working.index, dtype="float64")

    for _, group in working.groupby("Ticker", sort=False):
        values = []
        period_dates = list(group["Period End Date"])
        revenues = list(group["Total Revenue"])

        for index, revenue in enumerate(revenues):
            if index < 5:
                values.append(float("nan"))
                continue

            prior_revenue = revenues[index - 5]
            if revenue is None or prior_revenue is None or revenue <= 0 or prior_revenue <= 0:
                values.append(float("nan"))
                continue

            year_delta = (period_dates[index] - period_dates[index - 5]).days / 365.25
            if year_delta <= 0:
                values.append(float("nan"))
                continue

            values.append((revenue / prior_revenue) ** (1 / year_delta) - 1)

        working.loc[group.index, "Revenue CAGR 5-Year"] = values

    return working


def _ensure_seaborn_pandas_compat():
    import pandas as pd
    from pandas._config.config import OptionError, is_bool, register_option

    try:
        pd.get_option("mode.use_inf_as_null")
    except OptionError:
        register_option("mode.use_inf_as_null", False, validator=is_bool)


def _format_percent(value):
    import pandas as pd

    if pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def _format_ratio(value):
    import pandas as pd

    if pd.isna(value):
        return "n/a"
    return f"{float(value):.2f}x"


def _benchmark_direction(target_value, peer_value, higher_is_better=True):
    import pandas as pd

    if pd.isna(target_value) or pd.isna(peer_value):
        return "unavailable", 0

    if higher_is_better:
        if target_value > peer_value:
            return "above", 1
        if target_value < peer_value:
            return "below", -1
    else:
        if target_value < peer_value:
            return "better", 1
        if target_value > peer_value:
            return "worse", -1
    return "in line", 0


def _build_benchmark_summary_payload(
    ticker,
    sector,
    peer_tickers,
    latest_target_row,
    latest_peer_rows,
    trend_plot_path,
    ratio_plot_path,
):
    import pandas as pd

    peer_latest_revenue_yoy = latest_peer_rows["Revenue Growth YoY"].dropna().mean() if not latest_peer_rows.empty else float("nan")
    peer_latest_revenue_cagr = latest_peer_rows["Revenue CAGR 5-Year"].dropna().mean() if not latest_peer_rows.empty else float("nan")
    peer_latest_roe = latest_peer_rows["Return on Equity (ROE)"].dropna().mean() if not latest_peer_rows.empty else float("nan")
    peer_latest_gross_margin = latest_peer_rows["Gross Margin"].dropna().mean() if not latest_peer_rows.empty else float("nan")
    peer_latest_operating_margin = latest_peer_rows["Operating Margin"].dropna().mean() if not latest_peer_rows.empty else float("nan")
    peer_latest_net_margin = latest_peer_rows["Net Profit Margin"].dropna().mean() if not latest_peer_rows.empty else float("nan")
    peer_latest_de = latest_peer_rows["Debt-to-Equity (D/E)"].dropna().mean() if not latest_peer_rows.empty else float("nan")

    latest_revenue_yoy = latest_target_row.get("Revenue Growth YoY")
    latest_revenue_cagr = latest_target_row.get("Revenue CAGR 5-Year")
    latest_roe = latest_target_row.get("Return on Equity (ROE)")
    latest_gross_margin = latest_target_row.get("Gross Margin")
    latest_operating_margin = latest_target_row.get("Operating Margin")
    latest_net_margin = latest_target_row.get("Net Profit Margin")
    latest_de = latest_target_row.get("Debt-to-Equity (D/E)")

    comparisons = [
        ("Revenue YoY growth", latest_revenue_yoy, peer_latest_revenue_yoy, True),
        ("Revenue CAGR 5-Year", latest_revenue_cagr, peer_latest_revenue_cagr, True),
        ("ROE", latest_roe, peer_latest_roe, True),
        ("Gross margin", latest_gross_margin, peer_latest_gross_margin, True),
        ("Operating margin", latest_operating_margin, peer_latest_operating_margin, True),
        ("Net margin", latest_net_margin, peer_latest_net_margin, True),
        ("Debt / equity", latest_de, peer_latest_de, False),
    ]

    score = 0
    comparison_lines = []
    for label, target_value, peer_value, higher_is_better in comparisons:
        direction, delta = _benchmark_direction(target_value, peer_value, higher_is_better=higher_is_better)
        score += delta
        if "margin" in label.lower() or label == "ROE" or "Revenue" in label:
            comparison_lines.append(
                f"{label}: {ticker} {_format_percent(target_value)} vs {sector} peer average {_format_percent(peer_value)} ({direction})."
            )
        else:
            comparison_lines.append(
                f"{label}: {ticker} {_format_ratio(target_value)} vs {sector} peer average {_format_ratio(peer_value)} ({direction})."
            )

    if score >= 3:
        conclusion = (
            f"{ticker} screens as stronger than the current {sector} peer average on this benchmark set, "
            "with more growth/profitability wins than leverage weaknesses."
        )
    elif score <= -3:
        conclusion = (
            f"{ticker} screens weaker than the current {sector} peer average on this benchmark set, "
            "with more benchmark shortfalls than relative strengths."
        )
    else:
        conclusion = (
            f"{ticker} looks mixed versus the current {sector} peer average, with the benchmark picture split "
            "between strengths and weaker areas."
        )

    summary = "\n".join(
        [
            f"SQL benchmark analysis for {ticker} in {sector}.",
            f"Peers used: {', '.join(peer_tickers)}.",
            f"Latest annual period: {latest_target_row.get('Period End Date')}.",
            *comparison_lines,
            f"Trend plot: {trend_plot_path}.",
            f"Ratio plot: {ratio_plot_path}.",
        ]
    )

    return summary, conclusion


def _analysis_graph_documents(
    ticker,
    sector,
    peer_tickers,
    summary,
    conclusion,
    trend_plot_path,
    ratio_plot_path,
):
    from llama_index.core import Document

    shared_metadata = {
        "ticker": ticker,
        "sector": sector,
        "peer_tickers": peer_tickers,
        "source": "SQL benchmark analysis",
        "analysis_scope": "sector_peer_benchmark",
        "trend_plot_path": trend_plot_path,
        "ratio_plot_path": ratio_plot_path,
    }

    return [
        Document(
            text=summary,
            metadata={
                **shared_metadata,
                "type": "analysis_summary",
            },
        ),
        Document(
            text=conclusion,
            metadata={
                **shared_metadata,
                "type": "analysis_conclusion",
            },
        ),
    ]


def analyze_ticker_sql_benchmarks(
    ticker,
    peer_count=10,
    output_base_dir="plots",
    generate_plots=False,
    persist_to_graph=True,
):
    import os
    import pandas as pd

    ticker = str(ticker).upper()
    annual_history = _load_annual_financial_history_from_sql()
    if annual_history.empty:
        raise ValueError("No annual financial indicator data is available in the stock SQL database.")

    latest_annual = (
        annual_history
        .sort_values(["Ticker", "Period End Date"])
        .groupby("Ticker", sort=False)
        .tail(1)
        .copy()
    )

    sector_frame = _build_sql_sector_snapshot_frame(latest_annual["Ticker"].tolist())
    latest_annual = latest_annual.merge(sector_frame, on="Ticker", how="left")
    latest_annual = latest_annual.dropna(subset=["Sector"]).copy()

    target_latest = latest_annual[latest_annual["Ticker"] == ticker]
    if target_latest.empty:
        raise ValueError(f"{ticker} does not have annual SQL data with a resolved sector yet.")

    target_sector = target_latest["Sector"].iloc[0]
    peer_latest = (
        latest_annual[latest_annual["Sector"] == target_sector]
        .sort_values("Current Market Capitalization", ascending=False, kind="mergesort")
        .head(peer_count)
        .copy()
    )

    if ticker not in peer_latest["Ticker"].tolist():
        peer_latest = pd.concat([peer_latest, target_latest], ignore_index=True)
        peer_latest = peer_latest.drop_duplicates(subset=["Ticker"], keep="first")

    peer_tickers = sorted(peer_latest["Ticker"].unique().tolist())
    comparison_history = annual_history[annual_history["Ticker"].isin(peer_tickers)].copy()
    comparison_history = _add_revenue_cagr_5yr(comparison_history)
    comparison_history["Revenue Growth YoY Pct"] = comparison_history["Revenue Growth YoY"] * 100
    comparison_history["Revenue CAGR 5-Year Pct"] = comparison_history["Revenue CAGR 5-Year"] * 100

    target_history = comparison_history[comparison_history["Ticker"] == ticker].copy()
    peer_history = comparison_history[comparison_history["Ticker"] != ticker].copy()

    if target_history.empty:
        raise ValueError(f"{ticker} does not have annual SQL history available for plotting.")

    peer_trend_frames = []
    for metric_column in ("Revenue Growth YoY Pct", "Revenue CAGR 5-Year Pct"):
        metric_peer = peer_history.dropna(subset=[metric_column]).copy()
        if metric_peer.empty:
            continue
        aggregated = (
            metric_peer.groupby("Fiscal Year")[metric_column]
            .agg(peer_average="mean", peer_p25=lambda x: x.quantile(0.25), peer_p75=lambda x: x.quantile(0.75))
            .reset_index()
        )
        aggregated["Metric"] = metric_column
        peer_trend_frames.append(aggregated)

    peer_trends = pd.concat(peer_trend_frames, ignore_index=True) if peer_trend_frames else pd.DataFrame()

    ratio_metric_specs = [
        ("Return on Equity (ROE)", "ROE", True),
        ("Gross Margin", "Gross Margin", True),
        ("Operating Margin", "Operating Margin", True),
        ("Net Profit Margin", "Net Margin", True),
        ("Debt-to-Equity (D/E)", "Debt / Equity", False),
    ]

    latest_comparison = (
        comparison_history
        .sort_values(["Ticker", "Period End Date"])
        .groupby("Ticker", sort=False)
        .tail(1)
        .copy()
    )
    latest_comparison = _add_revenue_cagr_5yr(latest_comparison)
    latest_target_row = latest_comparison[latest_comparison["Ticker"] == ticker].iloc[0]
    latest_peer_rows = latest_comparison[latest_comparison["Ticker"] != ticker].copy()

    ratio_rows = []
    for column_name, label, is_percent in ratio_metric_specs:
        target_value = latest_target_row.get(column_name)
        peer_average = latest_peer_rows[column_name].dropna().mean() if not latest_peer_rows.empty else None

        if pd.notna(target_value):
            ratio_rows.append(
                {
                    "Metric": label,
                    "Series": ticker,
                    "Value": float(target_value) * (100 if is_percent else 1),
                    "Category": "percent" if is_percent else "ratio",
                }
            )
        if pd.notna(peer_average):
            ratio_rows.append(
                {
                    "Metric": label,
                    "Series": f"{target_sector} average",
                    "Value": float(peer_average) * (100 if is_percent else 1),
                    "Category": "percent" if is_percent else "ratio",
                }
            )

    ratio_frame = pd.DataFrame(ratio_rows)

    output_dir = os.path.join(output_base_dir, ticker)
    os.makedirs(output_dir, exist_ok=True)

    trend_path = os.path.join(output_dir, f"{ticker}_revenue_trends.png")
    ratio_path = os.path.join(output_base_dir, ticker, f"{ticker}_key_ratio_benchmark.png")

    if generate_plots:
        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt
        import seaborn as sns

        _ensure_seaborn_pandas_compat()
        sns.set_theme(style="whitegrid", context="talk")

        fig, axes = plt.subplots(2, 1, figsize=(14, 11), sharex=True)
        metric_specs = [
            ("Revenue Growth YoY Pct", "Revenue YoY Growth vs Same-Sector Peers", "Revenue YoY Growth (%)"),
            ("Revenue CAGR 5-Year Pct", "Revenue CAGR 5-Year vs Same-Sector Peers", "Revenue CAGR 5-Year (%)"),
        ]

        for axis, (metric_column, title, ylabel) in zip(axes, metric_specs):
            target_metric = target_history.dropna(subset=[metric_column]).copy()
            if not target_metric.empty:
                sns.lineplot(
                    data=target_metric,
                    x="Fiscal Year",
                    y=metric_column,
                    marker="o",
                    linewidth=2.5,
                    label=ticker,
                    ax=axis,
                )

            metric_peer = peer_trends[peer_trends["Metric"] == metric_column].copy()
            if not metric_peer.empty:
                axis.fill_between(
                    metric_peer["Fiscal Year"],
                    metric_peer["peer_p25"],
                    metric_peer["peer_p75"],
                    alpha=0.18,
                    color="#4c72b0",
                    label=f"{target_sector} peer IQR",
                )
                sns.lineplot(
                    data=metric_peer,
                    x="Fiscal Year",
                    y="peer_average",
                    marker="o",
                    linewidth=2.5,
                    linestyle="--",
                    label=f"{target_sector} peer average",
                    ax=axis,
                )

            axis.set_title(title)
            axis.set_ylabel(ylabel)
            axis.legend(loc="best")

        axes[-1].set_xlabel("Fiscal Year")
        fig.suptitle(
            f"{ticker} Revenue Trend Benchmarking vs {target_sector} Peers in SQL",
            y=1.02,
            fontsize=18,
        )
        fig.tight_layout()
        fig.savefig(trend_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [4, 1.4]})

        percent_frame = ratio_frame[ratio_frame["Category"] == "percent"].copy()
        if not percent_frame.empty:
            sns.barplot(data=percent_frame, x="Metric", y="Value", hue="Series", ax=axes[0])
            axes[0].set_ylabel("Percent")
            axes[0].set_xlabel("")
            axes[0].tick_params(axis="x", rotation=25)
            axes[0].set_title(f"{ticker} ROE and Margin Benchmark vs {target_sector} Average")
        else:
            axes[0].set_axis_off()

        ratio_only_frame = ratio_frame[ratio_frame["Category"] == "ratio"].copy()
        if not ratio_only_frame.empty:
            sns.barplot(data=ratio_only_frame, x="Metric", y="Value", hue="Series", ax=axes[1])
            axes[1].set_ylabel("Ratio")
            axes[1].set_xlabel("")
            axes[1].tick_params(axis="x", rotation=25)
            axes[1].set_title("Debt / Equity")
        else:
            axes[1].set_axis_off()

        for axis in axes:
            if axis.has_data():
                axis.legend(loc="best")

        fig.tight_layout()
        fig.savefig(ratio_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
    else:
        trend_path = None
        ratio_path = None

    summary, conclusion = _build_benchmark_summary_payload(
        ticker=ticker,
        sector=target_sector,
        peer_tickers=peer_tickers,
        latest_target_row=latest_target_row,
        latest_peer_rows=latest_peer_rows,
        trend_plot_path=trend_path,
        ratio_plot_path=ratio_path,
    )

    if persist_to_graph:
        import ingest_graph

        ingest_graph.upsert_ticker_analysis_documents(
            ticker,
            _analysis_graph_documents(
                ticker=ticker,
                sector=target_sector,
                peer_tickers=peer_tickers,
                summary=summary,
                conclusion=conclusion,
                trend_plot_path=trend_path,
                ratio_plot_path=ratio_path,
            ),
        )

    return {
        "ticker": ticker,
        "sector": target_sector,
        "peer_tickers": peer_tickers,
        "trend_plot_path": trend_path,
        "ratio_plot_path": ratio_path,
        "summary": summary,
        "conclusion": conclusion,
    }


def plot_sql_trends_and_benchmarks(ticker, peer_count=10, output_base_dir="plots"):
    return analyze_ticker_sql_benchmarks(
        ticker=ticker,
        peer_count=peer_count,
        output_base_dir=output_base_dir,
        generate_plots=True,
        persist_to_graph=True,
    )
