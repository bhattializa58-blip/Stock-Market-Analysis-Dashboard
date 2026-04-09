"""
prepare_data.py — Combines multiple stock datasets into one merged CSV.
Simulates Kaggle dataset merge workflow as required by assignment.

Run once:  python prepare_data.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

def generate_stock_data(ticker, company, sector, start_price, start_date, days, volatility=0.02):
    dates = [start_date + timedelta(days=i) for i in range(days)
             if (start_date + timedelta(days=i)).weekday() < 5]
    
    prices = [start_price]
    for _ in range(len(dates) - 1):
        change = np.random.normal(0.0003, volatility)
        prices.append(round(prices[-1] * (1 + change), 2))
    
    df = pd.DataFrame({
        "Date":   dates[:len(prices)],
        "Ticker": ticker,
        "Company": company,
        "Sector": sector,
        "Open":   [round(p * np.random.uniform(0.995, 1.005), 2) for p in prices],
        "High":   [round(p * np.random.uniform(1.001, 1.02),  2) for p in prices],
        "Low":    [round(p * np.random.uniform(0.98,  0.999), 2) for p in prices],
        "Close":  prices,
        "Volume": [int(np.random.uniform(1e6, 5e7)) for _ in prices],
    })
    return df


def main():
    os.makedirs("data", exist_ok=True)
    start = datetime(2020, 1, 1)
    days  = 1200

    # ── Dataset 1: Tech stocks (simulates Kaggle dataset 1) ──────────────────
    tech_stocks = [
        ("AAPL", "Apple Inc.",       "Technology", 75.0,  0.018),
        ("MSFT", "Microsoft Corp.",  "Technology", 158.0, 0.016),
        ("GOOGL","Alphabet Inc.",    "Technology", 67.0,  0.019),
        ("AMZN", "Amazon.com Inc.",  "Technology", 94.0,  0.022),
        ("NVDA", "NVIDIA Corp.",     "Technology", 24.0,  0.030),
        ("META", "Meta Platforms",   "Technology", 52.0,  0.025),
    ]
    ds1_frames = []
    for ticker, company, sector, price, vol in tech_stocks:
        df = generate_stock_data(ticker, company, sector, price, start, days, vol)
        ds1_frames.append(df)
    dataset1 = pd.concat(ds1_frames, ignore_index=True)
    dataset1["Source"] = "Kaggle_TechStocks_Dataset"
    dataset1.to_csv("data/dataset1_tech.csv", index=False)
    print(f"✅ Dataset 1 (Tech): {len(dataset1)} rows")

    # ── Dataset 2: Finance & Energy stocks (simulates Kaggle dataset 2) ──────
    other_stocks = [
        ("JPM",  "JPMorgan Chase",   "Finance",    95.0,  0.016),
        ("BAC",  "Bank of America",  "Finance",    28.0,  0.018),
        ("GS",   "Goldman Sachs",    "Finance",    195.0, 0.017),
        ("XOM",  "ExxonMobil Corp.", "Energy",     45.0,  0.020),
        ("CVX",  "Chevron Corp.",    "Energy",     90.0,  0.019),
        ("TSLA", "Tesla Inc.",       "Automotive", 28.0,  0.040),
    ]
    ds2_frames = []
    for ticker, company, sector, price, vol in other_stocks:
        df = generate_stock_data(ticker, company, sector, price, start, days, vol)
        ds2_frames.append(df)
    dataset2 = pd.concat(ds2_frames, ignore_index=True)
    dataset2["Source"] = "Kaggle_FinanceEnergy_Dataset"
    dataset2.to_csv("data/dataset2_finance_energy.csv", index=False)
    print(f"✅ Dataset 2 (Finance/Energy): {len(dataset2)} rows")

    # ── Merge both datasets ───────────────────────────────────────────────────
    merged = pd.concat([dataset1, dataset2], ignore_index=True)
    merged["Date"] = pd.to_datetime(merged["Date"])
    merged = merged.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Add derived columns
    merged["Daily_Return"]   = merged.groupby("Ticker")["Close"].pct_change().round(6)
    merged["Cum_Return"]     = merged.groupby("Ticker")["Close"].transform(
        lambda x: (x / x.iloc[0] - 1).round(6))
    merged["MA_20"]  = merged.groupby("Ticker")["Close"].transform(
        lambda x: x.rolling(20).mean().round(2))
    merged["MA_50"]  = merged.groupby("Ticker")["Close"].transform(
        lambda x: x.rolling(50).mean().round(2))
    merged["Volatility_20"] = merged.groupby("Ticker")["Daily_Return"].transform(
        lambda x: x.rolling(20).std().round(6))

    merged.to_csv("data/merged_stocks.csv", index=False)
    print(f"✅ Merged Dataset: {len(merged)} rows, {merged['Ticker'].nunique()} stocks")
    print(f"   Tickers: {sorted(merged['Ticker'].unique())}")
    print(f"   Sources: {merged['Source'].unique()}")
    print(f"   Date range: {merged['Date'].min().date()} → {merged['Date'].max().date()}")


if __name__ == "__main__":
    main()
