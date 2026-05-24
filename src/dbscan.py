import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from kmeans import add_features, stack_features

class DBSCANAnomalyDetector:
    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        feature_columns=None,
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.feature_columns = feature_columns
        self.scaler: StandardScaler | None = None
        self.dbscan: DBSCAN | None = None
        self.core_samples_: np.ndarray | None = None
        self.nn: NearestNeighbors | None = None
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> "DBSCANAnomalyDetector":
        df = add_features(df)
        X, _ = stack_features(df, self.feature_columns)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = self.dbscan.fit_predict(X_scaled)

        core_indices = self.dbscan.core_sample_indices_
        self.core_samples_ = X_scaled[core_indices]

        if len(self.core_samples_) > 0:
            self.nn = NearestNeighbors(radius=self.eps).fit(self.core_samples_)
        self.fitted = True
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted or self.scaler is None:
            raise ValueError("DBSCANAnomalyDetector must be fitted before scoring.")

        df = add_features(df)
        X, index = stack_features(df, self.feature_columns)
        X_scaled = self.scaler.transform(X)

        if self.nn is None:
            anomaly_flag = np.ones(len(X_scaled), dtype=int)
        else:
            distances, _ = self.nn.radius_neighbors(X_scaled, return_distance=True)
            anomaly_flag = np.array([0 if len(row) else 1 for row in distances], dtype=int)

        scored = df.loc[index].copy()
        scored["anomaly_flag"] = anomaly_flag
        return scored

    def score_walkforward(self, train_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
        combined = pd.concat([train_df, new_df]).sort_values(["ticker", "date"])
        combined = add_features(combined)
        X_all, _ = stack_features(combined, self.feature_columns)
        X_scaled = self.scaler.transform(X_all)
        
        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(X_scaled)
        
        # only return labels for the new block
        new_df = add_features(new_df)
        X_new, index = stack_features(new_df, self.feature_columns)
        n_new = len(X_new)
        new_labels = labels[-n_new:]
        
        scored = new_df.loc[index].copy()
        scored["anomaly_flag"] = (new_labels == -1).astype(int)
        return scored

def plot_kdistance(df: pd.DataFrame, feature_columns=None, min_samples: int = 10):
    df = add_features(df)
    X, _ = stack_features(df, feature_columns)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)
    kth_distances = np.sort(distances[:, -1])[::-1]
    
    plt.figure(figsize=(8, 4))
    plt.plot(kth_distances)
    plt.title(f"k-distance graph (k={min_samples})")
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"{min_samples}-th nearest neighbor distance")
    plt.axhline(y=0.5, color='r', linestyle='--', label='candidate eps')
    plt.legend()
    plt.tight_layout()
    plt.show()

def tune_dbscan(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns=None,
    eps_values=None,
    min_samples_values=None,
    target_range=(0.02, 0.08),
) -> pd.DataFrame:
    if eps_values is None:
        eps_values = [0.3, 0.5, 0.7, 0.9, 1.1]
    if min_samples_values is None:
        min_samples_values = [3, 5, 8, 10]

    rows = []
    for eps in eps_values:
        for min_samples in min_samples_values:
            detector = DBSCANAnomalyDetector(
                eps=eps,
                min_samples=min_samples,
                feature_columns=feature_columns,
            )
            detector.fit(train_df)
            scored = detector.score(val_df)
            flag_rate = float(scored["anomaly_flag"].mean())
            rows.append(
                {
                    "eps": eps,
                    "min_samples": min_samples,
                    "flag_rate": flag_rate,
                    "in_target": target_range[0] <= flag_rate <= target_range[1],
                }
            )

    result = pd.DataFrame(rows)
    result["distance_from_target"] = np.abs(result["flag_rate"] - np.mean(target_range))
    result.sort_values(["in_target", "distance_from_target", "flag_rate"], ascending=[False, True, True], inplace=True)
    return result
