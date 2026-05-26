import pandas as pd
import numpy as np

from src.rule_based import add_anomaly

def add_market_ret(df):
    df = df.copy()
    market_ret = df.groupby("date")["ret"].mean().rename("market_ret")
    df = df.merge(market_ret, on="date", how="left")
    return df

def add_market_breadth(df):
    df = df.copy()
    tickers_positive = (df[df['ret'] > 0].groupby("date").size())
    total_tickers = df.groupby("date").size()
    market_breadth = (tickers_positive / total_tickers).fillna(0).rename("market_breadth")
    df = df.merge(market_breadth, on="date", how="left")
    return df

def add_market_anomaly_flag(df):
    df = df.copy()
    market_df = df[["date", "market_ret", "market_breadth"]].drop_duplicates("date").sort_values("date")
    
    market_df["abs_mkt_ret"] = market_df["market_ret"].abs()
    shifted = market_df["abs_mkt_ret"].shift(1)
    rolling_95th = shifted.rolling(63, min_periods=63).quantile(0.95)
    
    market_df["market_anomaly_flag"] = (
        (market_df["abs_mkt_ret"] > rolling_95th) |
        (market_df["market_breadth"] < 0.3)
    ).astype(int)
    
    market_df = market_df.drop(columns=["abs_mkt_ret"])
    
    df = df.merge(market_df[["date", "market_anomaly_flag"]], on="date", how="left")
    return df

def compute_market(df):
    df=df.copy()
    df = add_anomaly(df)
    df = add_market_ret(df)
    df = add_market_breadth(df)
    df = add_market_anomaly_flag(df)
    return df


df = pd.read_csv("/home/p3r5eus/Documents/Stock Market Anomaly Detection/Stratex/data/processed.csv")
df = compute_market(df)
print(df[df["date"] == "2020-02-27"][["ticker","date", "market_ret", "market_breadth", "market_anomaly_flag"]])