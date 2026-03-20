import ingest_stock
import ingest_macro
import query
# get tickers for sp500, nasdaq100, and dow
def get_index_tickers(index):
    import pandas as pd
    import requests
    from io import StringIO
    index = index.lower()

    if index.startswith('s'):   # sp500
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    elif index.startswith('d'): # dow
        url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
    elif index.startswith('n'): # nasdaq / nasdaq100
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
        if 'Symbol' in df.columns:
            ticker_col = 'Symbol'
        elif 'Ticker' in df.columns:
            ticker_col = 'Ticker'

        if ticker_col is not None:
            # Optional: filter only rows that look like tickers (1-5 uppercase letters + optional ./-)
            tickers = df[ticker_col].dropna().astype(str).str.strip()
            tickers = tickers[tickers.str.match(r'^[A-Z0-9.-]+$')]  # basic ticker-like filter
            tickers = tickers.str.replace('.', '-', regex=False).tolist()
            if len(tickers) > 20:  # rough sanity check
                return tickers

    # If we reach here → no suitable table found
    raise ValueError(f"No ticker column ('Symbol' or 'Ticker') found in any table for {index}")

def ingest_companies(tickers):
    for ticker in tickers:
        ingest_stock.refresh_ticker_data_and_index(ticker)
        print()

def ask_question(ticker,query_str):
    return query.analyze_company(ticker, query_str)

ticker = input('which company do you want to know about, give me the ticker: ')


if __name__ == "__main__":
<<<<<<< HEAD
=======
    ticker = "AAPL"
>>>>>>> addingfeatures
    print(
        query.analyze_company(
            ticker,
            "Summarize the company's financial position and the current macro environment.",
        )
    )
