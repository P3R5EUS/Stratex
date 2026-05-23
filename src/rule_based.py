from feature import compute_ret_z,compute_range_pct,compute_vol_z
import pandas as pd
import numpy as np

def compute(df):
    df = df.copy()
    df = compute_ret_z(df)
    df = compute_vol_z(df)
    df = compute_range_pct(df)
    return df

def build_why(row):
    reasons = []
    if abs(row["ret_z"]) > 2.5:
        reasons.append("|ret_z|>2.5")
    if row["vol_z"] > 2.5:
        reasons.append("vol_z>2.5")
    if row["range_pct"] > 95:
        reasons.append("range_pct>95")
    return "; ".join(reasons)


def add_anomaly(df):
    df = compute(df)
    
    df["anomaly_flag"] = (
        (df["ret_z"].abs() > 2.5) | (df["vol_z"] > 2.5) | (df["range_pct"] > 95)
    ).astype(int)

    df["type"] = ""
    df.loc[df["ret_z"] < -2.5, "type"] += "crash "
    df.loc[df["ret_z"] >  2.5, "type"] += "spike "
    df.loc[df["vol_z"] >  2.5, "type"] += "volume_shock "
    df["type"] = df["type"].str.strip().str.replace(" ", "+")

    df["why"] = df.apply(build_why, axis=1)

    df.loc[df["anomaly_flag"] == 0, "type"] = ""
    df.loc[df["anomaly_flag"] == 0, "why"] = ""

    return df

##TESTING 
# 
# df = pd.read_csv("/home/p3r5eus/Documents/Stock Market Anomaly Detection/Stratex/data/processed.csv")
# df = add_anomaly(df)

# target = df[df["date"] == "2020-02-27"]

# print(
#     target[
#         [
#             "ticker",
#             "ret",
#             "ret_z",
#             "vol_z",
#             "range_pct",
#             "anomaly_flag",
#             "type",
#             "why"
#         ]
#     ]
#     .sort_values(by="ret_z")
#     .to_string(index=False)
# )