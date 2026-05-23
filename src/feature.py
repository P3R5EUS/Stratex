import pandas as pd
import numpy as np
from scipy.stats import percentileofscore

def compute_ret_z(df, W=63):
    df = df.copy()
    df["ret"] = df.groupby("ticker")["adj close"].pct_change()
    
    shifted = df.groupby("ticker")["ret"].shift(1)
    
    roll_mean = shifted.groupby(df["ticker"]).rolling(W, min_periods=W).mean().reset_index(level=0, drop=True)
    roll_std  = shifted.groupby(df["ticker"]).rolling(W, min_periods=W).std().reset_index(level=0, drop=True)
    
    df["ret_z"] = (df["ret"] - roll_mean) / roll_std
    return df

def compute_vol_z(df, W=21):
    df = df.copy()
    df["logV"] = np.log(df["volume"])
    
    shifted = df.groupby("ticker")["logV"].shift(1)
    
    roll_mean = shifted.groupby(df["ticker"]).rolling(W, min_periods=W).mean().reset_index(level=0, drop=True)
    roll_std  = shifted.groupby(df["ticker"]).rolling(W, min_periods=W).std().reset_index(level=0, drop=True)
    
    df["vol_z"] = (df["logV"] - roll_mean) / roll_std
    return df

def rolling_percentile(series, W):
    result = pd.Series(np.nan, index=series.index)
    arr = series.values
    for i in range(W, len(arr)):
        window = arr[i-W:i]   # past W days — no leakage
        today  = arr[i]
        result.iloc[i] = percentileofscore(window, today)
    return result

def compute_range_pct(df, W=63):
    df = df.copy()
    df["pct"] = (df["high"] - df["low"]) / df["close"]
    
    df["range_pct"] = (
        df.groupby("ticker")["pct"]
        .apply(lambda x: rolling_percentile(x, W))
        .reset_index(level=0, drop=True)
    )
    return df


def get_scores_by_date(df, date):
    # ensure datetime
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    target_date = pd.Timestamp(date)

    result = df[df["date"] == target_date].copy()

    if result.empty:
        print(f"No data found for {date}")
        return

    # sort by most extreme return z-score
    result["abs_ret_z"] = result["ret_z"].abs()

    result = result.sort_values(
        by="abs_ret_z",
        ascending=False
    )

    cols = [
        "ticker",
        "ret",
        "ret_z",
        "vol_z",
        "range_pct"
    ]

    print(result[cols].to_string(index=False))


# TESTING FOR A DATE : IN COVID TIME FOR A CRASH
# df = pd.read_csv("/home/p3r5eus/Documents/Stock Market Anomaly Detection/Stratex/data/processed.csv")
# df = compute_ret_z(df)
# df = compute_vol_z(df)
# df = compute_range_pct(df)
# get_scores_by_date(df, "2020-02-27")
