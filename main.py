def get_index_constituents(index):
    import pandas as pd
    import requests
    from io import StringIO

    index = index.lower()

    if index.startswith("s"):
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    elif index.startswith("d"):
        url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
    elif index.startswith("n"):
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    else:
        raise ValueError(f"Unknown index prefix: {index}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    html_content = StringIO(response.text)
    tables = pd.read_html(html_content)

    for df in tables:
        ticker_col = None
        if "Symbol" in df.columns:
            ticker_col = "Symbol"
        elif "Ticker" in df.columns:
            ticker_col = "Ticker"

        if ticker_col is not None:
            normalized_df = df.copy()
            normalized_df["Ticker"] = (
                normalized_df[ticker_col]
                .dropna()
                .astype(str)
                .str.strip()
            )
            normalized_df = normalized_df[normalized_df["Ticker"].str.match(r"^[A-Z0-9.-]+$")]
            normalized_df["Ticker"] = normalized_df["Ticker"].str.replace(".", "-", regex=False)
            if len(normalized_df) > 20:
                return normalized_df.reset_index(drop=True)

    raise ValueError(f"No ticker column ('Symbol' or 'Ticker') found in any table for {index}")


def get_index_tickers(index):
    constituents = get_index_constituents(index)
    return constituents["Ticker"].dropna().astype(str).tolist()


def _coerce_market_cap(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _fetch_current_market_cap(ticker):
    import yfinance as yf

    stock = yf.Ticker(ticker)

    try:
        fast_info = stock.fast_info
    except Exception:
        fast_info = {}

    market_cap = _coerce_market_cap(fast_info.get("marketCap"))
    if market_cap is not None:
        return market_cap

    try:
        info = stock.get_info()
    except Exception:
        info = {}

    return _coerce_market_cap(info.get("marketCap"))


def _fetch_sector_and_market_cap_snapshot(ticker):
    import ingest_stock

    ticker_map = ingest_stock._load_sec_ticker_map()
    sec_identity = ticker_map.get(ticker)
    sic = None
    sic_description = None

    if sec_identity:
        cik = sec_identity["cik"]
        submissions = ingest_stock._sec_get_json(ingest_stock.SEC_SUBMISSIONS_URL_TEMPLATE.format(cik=cik))
        sic = ingest_stock._clean_profile_field(submissions.get("sic"))
        sic_description = ingest_stock._clean_profile_field(submissions.get("sicDescription"))

    return {
        "Ticker": ticker,
        "Industry": ingest_stock._derive_industry_from_sic(sic, sic_description),
        "Sector": ingest_stock._derive_sector_from_sic(sic, sic_description),
        "SIC": sic,
        "SIC Description": sic_description,
        "Market Capitalization": _fetch_current_market_cap(ticker),
    }


def get_top_market_cap_companies_by_sector(index="sp500", top_n=5, max_workers=12):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import pandas as pd

    constituents = get_index_constituents(index)
    columns_to_keep = ["Ticker"]
    if "Security" in constituents.columns:
        columns_to_keep.append("Security")

    ranked = constituents[columns_to_keep].dropna(subset=["Ticker"]).copy()
    ranked["Ticker"] = ranked["Ticker"].astype(str).str.strip()
    ranked = ranked.drop_duplicates(subset=["Ticker"])

    snapshots = []
    failures = []

    tickers = ranked["Ticker"].tolist()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_fetch_sector_and_market_cap_snapshot, ticker): ticker for ticker in tickers}
        for future in as_completed(future_map):
            ticker = future_map[future]
            try:
                snapshot = future.result()
            except Exception:
                snapshot = None
            if not snapshot or snapshot.get("Market Capitalization") is None or not snapshot.get("Industry"):
                failures.append(ticker)
                continue
            snapshots.append(snapshot)

    if not snapshots:
        raise ValueError(f"No market-cap sector snapshots were available for {index}.")

    snapshots_df = pd.DataFrame(snapshots)
    ranked = ranked.merge(snapshots_df, on="Ticker", how="inner")
    ranked = ranked.sort_values(
        ["Industry", "Market Capitalization", "Ticker"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    ranked = ranked.groupby("Industry", sort=True, group_keys=False).head(top_n).reset_index(drop=True)
    ranked.attrs["industry_column"] = "Industry"
    ranked.attrs["missing_market_cap_tickers"] = failures
    return ranked


def ingest_companies(tickers):
    import ingest_stock

    for ticker in tickers:
        ingest_stock.refresh_ticker_data_and_index(ticker)
        print()


def ingest_companies_to_database(tickers):
    import analysis
    import ingest_stock

    total = len(tickers)
    for index, ticker in enumerate(tickers, start=1):
        print(f"[{index}/{total}] Updating database for {ticker}...")
        try:
            ingest_stock.update_financial_records(ticker)
            analysis_result = analysis.get_or_create_daily_benchmark_analysis(
                ticker,
                generate_plots=False,
                persist_to_graph=True,
            )
            print(f"Stored benchmark analysis for {ticker}: {analysis_result['conclusion']}")
        except Exception as exc:
            print(f"Skipping {ticker}: {exc}")
        print()


def ingest_top_market_cap_companies_by_sector(index="sp500", top_n=5):
    selection = get_top_market_cap_companies_by_sector(index=index, top_n=top_n)
    industry_column = selection.attrs.get("industry_column", "Industry")

    print(
        f"Ingesting {len(selection)} tickers from the top {top_n} by market cap in each "
        f"{industry_column} bucket from {index.upper()}."
    )
    print()

    for industry, industry_df in selection.groupby(industry_column, sort=True):
        tickers = industry_df["Ticker"].tolist()
        ticker_line = ", ".join(tickers)
        print(f"{industry}: {ticker_line}")
        print()

    ingest_companies_to_database(selection["Ticker"].tolist())
    return selection


def ask_question(ticker, query_str):
    import query

    return query.analyze_company(ticker, query_str)


def _normalize_main_input(value):
    normalized = (value or "").strip().strip("\"'").lower()
    normalized = normalized.replace("_", "-")
    normalized = "-".join(normalized.split())
    while "--" in normalized:
        normalized = normalized.replace("--", "-")
    return normalized


def _extract_plot_ticker(value):
    normalized = _normalize_main_input(value)
    if not normalized.startswith("plot-"):
        return None
    ticker = normalized.split("-", 1)[1].strip().upper()
    return ticker or None


def _show_generated_plots(*plot_paths):
    import os

    shown_paths = []
    for plot_path in plot_paths:
        if not plot_path:
            continue
        absolute_plot_path = os.path.abspath(plot_path)
        if not os.path.isfile(absolute_plot_path):
            continue
        if hasattr(os, "startfile"):
            try:
                os.startfile(absolute_plot_path)
                shown_paths.append(absolute_plot_path)
            except OSError as exc:
                print(f"Could not open plot {absolute_plot_path}: {exc}")
    return shown_paths


def main():
    import asyncio
    import sys
    import time

    if sys.platform.startswith("win") and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    user_input = input(
        "which company do you want to know about, give me the ticker "
        "(or type 'top5-by-sector' to ingest the S&P 500 top 5 by market cap in each SEC SIC-derived industry, "
        "or 'plot AAPL' to generate SQL-based benchmark charts): "
    ).strip()
    normalized_input = _normalize_main_input(user_input)

    if normalized_input in {"top5-by-sector", "top5-by-industry"}:
        selection = ingest_top_market_cap_companies_by_sector(index="sp500", top_n=5)
        print()
        print(
            "Finished ingesting "
            f"{len(selection)} selected tickers grouped by {selection.attrs.get('industry_column', 'Industry')}."
        )
        return

    plot_ticker = _extract_plot_ticker(user_input)
    if plot_ticker:
        import analysis

        plot_result = analysis.plot_sql_trends_and_benchmarks(plot_ticker)
        print(plot_result["summary"])
        print()
        print(f"Conclusion: {plot_result['conclusion']}")
        print()
        print(
            f"Generated benchmark plots for {plot_result['ticker']} vs {plot_result['industry']} peers: "
            f"{plot_result['trend_plot_path']} and {plot_result['ratio_plot_path']}"
        )
        return

    ticker = user_input.strip().strip("\"'")

    import query

    start_time = time.perf_counter()
    result = query.analyze_company_package(
        ticker,
        "Summarize the company's financial position and the current macro environment.",
        include_plots=True,
    )
    elapsed_seconds = time.perf_counter() - start_time

    answer = result["answer"]
    analysis_result = result.get("analysis_result") or {}
    print(answer)
    if analysis_result:
        print()
        print("Analysis summary:")
        print(analysis_result.get("conclusion") or analysis_result.get("summary") or "No analysis summary was available.")

        shown_plots = _show_generated_plots(
            analysis_result.get("trend_plot_path"),
            analysis_result.get("ratio_plot_path"),
        )
        if shown_plots:
            print()
            print("Opened benchmark plots:")
            for plot_path in shown_plots:
                print(plot_path)
        else:
            generated_paths = [
                path
                for path in (
                    analysis_result.get("trend_plot_path"),
                    analysis_result.get("ratio_plot_path"),
                )
                if path
            ]
            if generated_paths:
                print()
                print("Generated benchmark plots:")
                for plot_path in generated_paths:
                    print(plot_path)
    print(f"\nAnswer generation time: {elapsed_seconds:.2f} seconds")


if __name__ == "__main__":
    main()
