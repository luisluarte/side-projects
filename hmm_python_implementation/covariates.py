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

#--- feature endomorphisms ---

def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    print("adding log returns")
    df = df.copy(); df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1)); return df

def add_smoothed_log_returns(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    print("adding smooth returns")
    df = df.copy(); df['obs_smoothed'] = df['log_ret'] .rolling(window=window).mean(); return df

def add_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    print("adding volatility")
    df = df.copy(); df['z_vol'] = df['log_ret'].rolling(window=window).std(); return df

def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    print("adding rsi")
    df = df.copy(); df['z_rsi'] = df.ta.rsi(close='Close', length=window)/100.0; return df

def add_obv_roc(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    print("adding roc")
    df = df.copy(); obv = df.ta.obv(close='Close', volume='Volume'); df['z_obv_roc'] = obv.pct_change(periods=window).replace([np.inf, -np.inf], 0.0); return df

def add_vix_norm(df: pd.DataFrame) -> pd.DataFrame:
    print("vix only")
    vix_col = 'z_IND_VIX'
    if vix_col in df.columns:
        df['z_vix'] = df[vix_col] / 100.0
    else:
        df['z_vix'] = pd.Series(pd.NA, dtype='Int64', index=df.index)
    return df

def add_dxy_returns(df: pd.DataFrame) -> pd.DataFrame:
    print("dxy only")
    dxy_col = 'z_DX_Y_NYB'
    if dxy_col in df.columns:
        df['z_dxy_ret'] = np.log(df[dxy_col] / df[dxy_col].shift(1))
    else:
        df['z_dxy_ret'] = pd.Series(pd.NA, dtype='Int64', index=df.index)
    return df


# --- source morphism ---
def fetch_market_data(
    asset_config: Dict,
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

    # split the json file in two variables
    tickers = [d['symbol'] for d in asset_config if 'symbol' in d]
    sector = [s['sector'] for s in asset_config if 'sector' in s]

    for index, ticker in enumerate(tickers):
        try:
            print(f"--- fetching {ticker} ---")
            raw_df = yf.download(
                ticker, start=start_date, end=end_date, progress=False, auto_adjust = True
            )
            print(raw_df)
            if isinstance(raw_df.columns, pd.MultiIndex):
                raw_df.columns = raw_df.columns.get_level_values(0)

            raw_df['ticker'] = tickers[index]
            raw_df['sector'] = sector[index]

            # remove column with row index
            raw_df.reset_index(inplace=True)

            # add features
            f1 = add_log_returns(raw_df)
            f2 = add_smoothed_log_returns(f1)
            f3 = add_volatility(f2)
            f4 = add_rsi(f3)
            f5 = add_obv_roc(f4)
            f6 = add_vix_norm(f5)
            f7 = add_dxy_returns(f6)
            final_df = f7
            
            file_path = str(folder_path) + "/" + ticker + ".csv"
            print(f"--- persisting data to: {file_path} ---")
            final_df.to_csv(file_path, index=False)
        except Exception as e:
            print(f"failed to download for {ticker}: {e}")
        time.sleep(1)

    return raw_df
