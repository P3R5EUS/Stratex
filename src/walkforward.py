import pandas as pd
import numpy as np
from dbscan import DBSCANDetector
from kmeans import FEATURES, KMEANSDetector
from market import compute_market
import os

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
    # 1. load + prep
    full_df = pd.read_csv("/home/p3r5eus/Documents/Stock Market Anomaly Detection/Stratex/data/processed.csv")
    full_df["date"] = pd.to_datetime(full_df["date"])
    full_df = compute_market(full_df)

    train_df = full_df[full_df["date"].dt.year == 2018].copy()

    # 2. fit kmeans
    # hint: use find_best_k to pick k, or hardcode after tuning
    kmeans_detector = KMEANSDetector(k=3, q=95)
    kmeans_detector.fit(train_df.dropna(subset=FEATURES))

    # 3. fit dbscan scaler
    dbscan_detector = DBSCANDetector(eps=0.5, min_samples=10)
    dbscan_detector.fit_scaler(train_df.dropna(subset=FEATURES))

    # 4. run
    final_df = run_walkforward(full_df, kmeans_detector, dbscan_detector)

    # 5. save
    save_outputs(final_df)
    print("Done — outputs written.")