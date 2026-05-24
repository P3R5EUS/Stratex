from typing import Union
import pandas as pd
from dbscan import DBSCANAnomalyDetector
from kmeans import KMeansAnomalyDetector


def monthly_walkforward(
    df: pd.DataFrame,
    detector: Union[KMeansAnomalyDetector,DBSCANAnomalyDetector],
    initial_train_end: str,
    final_score_month: str,
) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["period"] = df["date"].dt.to_period("M")

    final_period = pd.Period(final_score_month, "M")
    current_period = pd.Period(initial_train_end, "M") + 1

    scored_blocks = []
    while current_period <= final_period:
        train_mask = df["period"] <= current_period - 1
        score_mask = df["period"] == current_period

        train_df = df.loc[train_mask].copy()
        score_df = df.loc[score_mask].copy()

        if train_df.empty or score_df.empty:
            current_period += 1
            continue
        
        detector.fit(train_df)
        if hasattr(detector, "score_walkforward"):
            scored_block = detector.score_walkforward(train_df, score_df)
        else:
            scored_block = detector.score(score_df)

        scored_block["walkforward_period"] = current_period.strftime("%Y-%m")
        scored_blocks.append(scored_block)

        current_period += 1

    if scored_blocks:
        result = pd.concat(scored_blocks, ignore_index=True)
    else:
        result = pd.DataFrame()
    return result


def summarize_flag_rates(scored_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        scored_df.groupby("walkforward_period")
        .agg(
            flag_rate=("anomaly_flag", "mean"),
            flagged_count=("anomaly_flag", "sum"),
            total_count=("anomaly_flag", "size"),
        )
        .reset_index()
    )
    return summary


