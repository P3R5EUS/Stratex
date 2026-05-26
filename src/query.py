'''
input :
python -m src.query --date 2020-02-27

output :
=== Market Status: 2020-02-27 ===
Market Return : -0.069
Breadth       : 0.0
Market Flag   : ANOMALOUS

=== Anomalous Tickers ===
Ticker  ret_z   vol_z   range_pct   type                why
AAPL    -4.13   2.68    96.8        crash+volume_shock  |ret_z|>2.5; vol_z>2.5
MSFT    -5.32   2.87    98.4        crash               |ret_z|>2.5
...
'''

# query.py
import argparse
import pandas as pd

def query_date(date_str, anomaly_card_path, market_days_path):
    # 1. load both CSVs
    anomaly_card = pd.read_csv(anomaly_card_path)
    market_days = pd.read_csv(market_days_path)
        # convert date columns to datetime for filtering
    anomaly_card["date"] = pd.to_datetime(anomaly_card["date"])
    market_days["date"] = pd.to_datetime(market_days["date"])

    # 2. filter anomaly_card to given date
    #    → show all rows where anomaly_flag == 1
    anomaly_card = anomaly_card[anomaly_card["date"] == pd.to_datetime(date_str)]
    # 3. filter market_days to given date
    #    → show market_ret, breadth, market_anomaly_flag
    market_days = market_days[market_days["date"] == pd.to_datetime(date_str)]

    # 4. print market status section
    print(f"=== Market Status: {date_str} ===")
    print(f"Market Return : {market_days['market_ret'].values[0]:.3f}")
    print(f"Breadth       : {market_days['market_breadth'].values[0]:.1f}")
    print(f"Market Flag   : {'ANOMALOUS' if market_days['market_anomaly_flag'].values[0] == 1 else 'NORMAL'}")
    print("\n=== Anomalous Tickers ===")
    if anomaly_card.empty:
        print("No anomalies detected")
    else:
        display_cols = ["ticker", "ret_z", "vol_z", "range_pct", "type", "why"]
        print(anomaly_card[display_cols].to_string(index=False))
    # 5. print anomalous tickers section
    #    if no anomalies → print "No anomalies detected"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    
    query_date(args.date, 
               "/home/p3r5eus/Documents/Stock Market Anomaly Detection/Stratex/output/anomaly_cards/anomaly_card.csv",
               "/home/p3r5eus/Documents/Stock Market Anomaly Detection/Stratex/output/market_days/market_days.csv")