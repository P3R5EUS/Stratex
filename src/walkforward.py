import pandas as pd
import numpy as np
from src.dbscan import DBSCANDetector
from src.kmeans import FEATURES, KMEANSDetector
from src.market import compute_market
import os
import argparse

def run_walkforward(full_df, kmeans_detector, dbscan_detector):
    # 1. split
    train = full_df[full_df["date"].dt.year == 2018]
    val   = full_df[full_df["date"].dt.year == 2019]
    test  = full_df[(full_df["date"].dt.year == 2020) & 
                    (full_df["date"].dt.month <= 3)]

    # 2. KMeans — score val and test directly
    val_km  = kmeans_detector.score(val)
    test_km = kmeans_detector.score(test)
    km_scored = pd.concat([val_km, test_km])

    # 3. DBSCAN — monthly walk-forward
    history = train.copy()
    results = []

    for period in [val, test]:
        for _, block in period.groupby(period["date"].dt.to_period("M")):
            scored = dbscan_detector.score_block(history, block)
            results.append(scored)
            history = pd.concat([history, block])

    db_scored = pd.concat(results)

    # 4. merge kmeans_flag onto db_scored on (date, ticker)
    final_df = db_scored.merge(
        km_scored[["date", "ticker", "kmeans_flag"]],
        on=["date", "ticker"], how="left"
    )
    final_df["kmeans_flag"] = final_df["kmeans_flag"].fillna(0).astype(int)

    
    # 5. consensus
    # union        = kmeans_flag | dbscan_flag
    # intersection = kmeans_flag & dbscan_flag
    final_df["anomaly_flag"] = ((final_df["kmeans_flag"] == 1) | (final_df["dbscan_flag"] == 1)).astype(int)    

    # 6. return final_df
    return final_df

def save_outputs(final_df):
    """
    Writes two CSVs to outputs/ folder:
    - anomaly_cards/anomaly_card.csv
    - market_days/market_days.csv
    """
    anomaly_card = final_df[final_df["anomaly_flag"] == 1][
    ["date", "ticker", "anomaly_flag", "type", 
     "ret", "ret_z", "vol_z", "range_pct", "why"]
    ].copy()
    os.makedirs("/home/p3r5eus/Documents/Stock Market Anomaly Detection/Stratex/output/anomaly_cards/", exist_ok=True)
    anomaly_card.to_csv("/home/p3r5eus/Documents/Stock Market Anomaly Detection/Stratex/output/anomaly_cards/anomaly_card.csv", index=False)

    os.makedirs("/home/p3r5eus/Documents/Stock Market Anomaly Detection/Stratex/output/market_days/", exist_ok=True)
    market_days = final_df[["date", "market_ret", "market_breadth", "market_anomaly_flag"]].drop_duplicates(subset=["date"]).copy()
    market_days.to_csv("/home/p3r5eus/Documents/Stock Market Anomaly Detection/Stratex/output/market_days/market_days.csv", index=False)

if __name__ == "__main__":

    '''
    for input : python -m src.walkforward --universe QQQ,AAPL,MSFT,NVDA,AMZN,TSLA

    we will output : 2 csvs in the output folder:
    - output/anomaly_cards/anomaly_card.csv
    - output/market_days/market_days.csv
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe",type = str,default="QQQ,AAPL,MSFT,NVDA,AMZN,TSLA")
    args = parser.parse_args()

    tickers = [t.strip() for t in args.universe.split(",")]
    full_df = pd.read_csv("/home/p3r5eus/Documents/Stock Market Anomaly Detection/Stratex/data/processed.csv")
    full_df = compute_market(full_df)
    full_df["date"] = pd.to_datetime(full_df["date"])
    full_df = full_df[full_df["ticker"].isin(tickers)].copy()

    train_df = full_df[full_df["date"].dt.year == 2018].copy()

    kmeans_detector = KMEANSDetector(k=3, q=95)   
    kmeans_detector.fit(train_df.dropna(subset=FEATURES))

    dbscan_detector = DBSCANDetector(eps=0.5, min_samples=10)
    dbscan_detector.fit_scaler(train_df.dropna(subset=FEATURES))

    final_df = run_walkforward(full_df, kmeans_detector, dbscan_detector)

    save_outputs(final_df)
    print("Walk-forward complete. Outputs saved to output/ folder.")