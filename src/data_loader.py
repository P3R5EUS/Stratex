import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#define UNIVERSE and extract data accordingly
UNIVERSE = ["QQQ", "AAPL", "MSFT", "AMZN", "NVDA", "TSLA"]

def load_universe(data_dir,universe):
    dfs = []
    for ticker in universe:
        path = os.path.join(data_dir, f"{ticker}.csv")
        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.rename(columns=str.lower)  # normalize column names
        df["ticker"] = ticker
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    # #all null values have been erased
    # combined.dropna(subset=["open"],inplace=True)
    
    return combined

df = load_universe("/home/p3r5eus/Documents/Stock Market Anomaly Detection/Stratex/data/stocks",UNIVERSE)

df.info()