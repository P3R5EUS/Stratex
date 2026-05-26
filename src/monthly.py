'''
input : python -m src.monthly --month 2020-02
output:
========================================
Monthly Report: 2020-02
========================================
Date Ticker Type ret_z vol_z market_anomaly_flag   why
....
....
'''
# monthly.py
import argparse
import pandas as pd

def monthly_report(month_str, market_days_path, anomaly_card_path):
    # 1. load both CSVs
    market_days = pd.read_csv(market_days_path)
    anomaly_card = pd.read_csv(anomaly_card_path)

    # 2. filter market_days to given month
    market_days["date"] = pd.to_datetime(market_days["date"])
    month_start = pd.to_datetime(month_str + "-01")
    month_end = month_start + pd.offsets.MonthEnd(0)
    monthly_market = market_days[(market_days["date"] >= month_start) & (market_days["date"] <= month_end)]

    # 3. filter anomaly_card to given month
    anomaly_card["date"] = pd.to_datetime(anomaly_card["date"])
    monthly_anomalies = anomaly_card[(anomaly_card["date"] >= month_start) & (anomaly_card["date"] <= month_end)]

    # 4. print report
    print("========================================")
    print(f"Monthly Report: {month_str}")
    print("========================================")
    if monthly_anomalies.empty:
        print("No anomalies detected in this month.")
    else:
        # merge market_anomaly_flag onto anomalies
        monthly_anomalies = monthly_anomalies.merge(
            monthly_market[["date", "market_anomaly_flag"]],
            on="date", how="left"
        )

        # then display
        display_cols = ["date", "ticker", "type", "ret_z", "vol_z", "market_anomaly_flag", "why"]
        print(monthly_anomalies[display_cols].to_string(index=False))
    
        # after the table — useful for the report
    print(f"\nTotal anomalous days : {monthly_anomalies['date'].nunique()}")
    print(f"Total flagged rows   : {len(monthly_anomalies)}")
    worst = monthly_market.loc[monthly_market["market_ret"].idxmin()]
    print(f"Worst market day     : {worst['date'].date()} ({worst['market_ret']:.3f})")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", required=True, help="YYYY-MM")
    args = parser.parse_args()
    
    monthly_report(args.month, 
                   "/home/p3r5eus/Documents/Stock Market Anomaly Detection/Stratex/output/market_days/market_days.csv",
                   "/home/p3r5eus/Documents/Stock Market Anomaly Detection/Stratex/output/anomaly_cards/anomaly_card.csv")
    