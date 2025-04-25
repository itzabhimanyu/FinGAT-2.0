import os
import pickle
import pandas as pd
from tqdm import tqdm
import yfinance as yf

def fetch_and_store(symbols, start, end, out_dir="data"):
    """
    Downloads OHLC history for each symbol and:
      • writes data/<SYMBOL>.csv
      • pickles the full dict to data/all_data.pkl
    """
    os.makedirs(out_dir, exist_ok=True)
    all_data = {}
    for s in tqdm(symbols, desc="Downloading raw OHLC"):
        try:
            df = yf.download(f"{s}.NS", start=start, end=end, progress=False)
        except Exception:
            continue
        if df.empty or len(df) < 2:
            continue
        df.to_csv(os.path.join(out_dir, f"{s}.csv"))
        all_data[s] = df

    with open(os.path.join(out_dir, "all_data.pkl"), "wb") as f:
        pickle.dump(all_data, f)

    print(f"✅ Downloaded {len(all_data)} symbols → '{out_dir}/' + all_data.pkl")


if __name__ == "__main__":
    # 1. Read your Nifty500 CSV
    nifty = pd.read_csv("ind_nifty500list_filtered_final.csv")
    symbols = (
        nifty["Symbol"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    # 2. Run download for all symbols
    fetch_and_store(
        symbols,
        start="2022-01-10",
        end="2025-01-10",
        out_dir="data"
    )
