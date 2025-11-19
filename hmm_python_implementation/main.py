# libs
import os
import torch
import sys
import argparse
from covariates import *


def main():
    # set main directory
    script_locus = os.path.abspath(__file__)
    script_context = os.path.dirname(script_locus)
    os.chdir(script_context)
    print(f"--- context normalized to {script_context} ---")

    # terminal arguments
    parser = argparse.ArgumentParser(description="run hmm code")
    parser.add_argument(
        "start_date", type=str, default="2018-01-01", help="set start date YYYY-MM-DD"
    )
    parser.add_argument(
        "fetch_market_data",
        type=str,
        default="y",
        choices=["y", "n"],
        help="download market data?",
    )
    args = parser.parse_args()

    # --- read tickers ---
    asset_config = parse_tickers(filepath = 'tickers.json')
    print("--- getting tickers ---")
    print(asset_config)

    # --- download raw data ---
    print("--- downloading market data")
    if args.fetch_market_data == 'y':
        market_data = fetch_market_data(
            asset_config=asset_config,
            start_date="2018-01-01",
            folder="market_data",
            fetch=args.fetch_market_data.lower()
        )


if __name__ == "__main__":
    main()
