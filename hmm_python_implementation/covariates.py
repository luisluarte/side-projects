# libs
import os
import time
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import torch
import numpy as np
import json
from functools import reduce, partial
from datetime import datetime
from typing import List, Tuple, Dict
from pathlib import Path


# --- parse ticker json ---
def parse_tickers(filepath: str = "tickers.json") -> Dict:
    """
    lifts the external json state into a python dictionary
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"configuration file not found: {filepath}")

    with open(filepath, "r") as f:
        config = json.load(f)

    if "target" not in config or "covariates" not in config:
        raise ValueError("config must contain target and covariates keys.")

    print(config["covariates"])

    return config["covariates"]


# --- source morphism ---
def fetch_market_data(
    tickers: List[str],
    folder: str,
    start_date: str,
    end_date: str = None,
    fetch: str = "y",
) -> pd.DataFrame:
    """
    (world state) -> raw data frame
    fetches OHLCV data for a list of tickers
    """
    if end_date is None:
        end_data = datetime.today().strftime("%Y-%m-%d")

    folder_path = Path(folder)

    if not os.path.exists(folder):
        print(f"created directory: {folder}")
        folder_path.mkdir(exist_ok=True)

    print(f"--- fetching {tickers} ---")
    for ticker in tickers:
        try:
            raw_df = yf.download(
                ticker, start=start_date, end=end_date, progress=False, auto_adjust = True
            )
            if isinstance(raw_df.columns, pd.MultiIndex):
                raw_df.columns = raw_df.columns.get_level_values(0)

            raw_df['ticker'] = ticker
            raw_df.reset_index(inplace=True)
            file_path = str(folder_path) + "/" + ticker + ".csv"
            print(f"--- persisting data to: {file_path} ---")
            raw_df.to_csv(file_path, index=False)
        except Exception as e:
            print(f"failed to download for {ticker}: {e}")
        time.sleep(1)

    return raw_df
