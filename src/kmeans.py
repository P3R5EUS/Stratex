import os
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from feature import compute_range_pct, compute_ret_z, compute_vol_z

DEFAULT_FEATURE_COLUMNS = ["ret_z", "vol_z", "range_pct"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "ret" not in df.columns:
        df["ret"] = df.groupby("ticker")["adj close"].pct_change()
    df = compute_ret_z(df)
    df = compute_vol_z(df)
    df = compute_range_pct(df)
    return df


def stack_features(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
) -> Tuple[np.ndarray, pd.Index]:
    if feature_columns is None:
        feature_columns = DEFAULT_FEATURE_COLUMNS
    X = df[feature_columns].copy()
    X = X.dropna()
    return X.values, X.index


def load_processed_data(path: str = "data/processed.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def train_val_test_split(
    df: pd.DataFrame,
    train_years: Iterable[int],
    val_years: Iterable[int],
    test_years: Iterable[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    train = df[df["date"].dt.year.isin(train_years)].copy()
    val = df[df["date"].dt.year.isin(val_years)].copy()
    test = df[df["date"].dt.year.isin(test_years)].copy()
    return train, val, test


class KMeansAnomalyDetector:
    def __init__(
        self,
        n_clusters: int = 4,
        q: float = 97.5,
        random_state: int = 42,
        feature_columns: Optional[List[str]] = None,
    ):
        self.n_clusters = n_clusters
        self.q = q
        self.random_state = random_state
        self.feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS
        self.scaler: Optional[StandardScaler] = None
        self.kmeans: Optional[KMeans] = None
        self.thresholds: Optional[Dict[int, float]] = None
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> "KMeansAnomalyDetector":
        df = add_features(df)
        X, _ = stack_features(df, self.feature_columns)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init="auto",
        )
        self.kmeans.fit(X_scaled)

        self.thresholds = self._compute_thresholds(X_scaled, self.kmeans.labels_, self.q)
        self.fitted = True
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted or self.kmeans is None or self.scaler is None or self.thresholds is None:
            raise ValueError("KMeansAnomalyDetector must be fitted before scoring.")

        df = add_features(df)
        X, index = stack_features(df, self.feature_columns)
        X_scaled = self.scaler.transform(X)
        labels = self.kmeans.predict(X_scaled)
        distances = self._distance_to_centroid(X_scaled, labels)
        thresholds = np.array([self.thresholds[label] for label in labels])

        scored = df.loc[index].copy()
        scored["cluster"] = labels
        scored["distance"] = distances
        scored["threshold"] = thresholds
        scored["anomaly_flag"] = (distances > thresholds).astype(int)
        return scored

    def set_threshold_quantile(self, q: float) -> None:
        if self.kmeans is None or self.scaler is None:
            raise ValueError("Fit the KMeansAnomalyDetector before setting a threshold quantile.")
        self.q = q

    def _distance_to_centroid(self, X_scaled: np.ndarray, labels: np.ndarray) -> np.ndarray:
        centers = self.kmeans.cluster_centers_
        return np.linalg.norm(X_scaled - centers[labels], axis=1)

    def _compute_thresholds(
        self, X_scaled: np.ndarray, labels: np.ndarray, q: float
    ) -> Dict[int, float]:
        distances = self._distance_to_centroid(X_scaled, labels)
        thresholds: Dict[int, float] = {}
        for label in np.unique(labels):
            cluster_distances = distances[labels == label]
            thresholds[int(label)] = float(np.percentile(cluster_distances, q))
        return thresholds


def evaluate_k_range(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    k_values: Optional[Iterable[int]] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    df = add_features(df)
    X, _ = stack_features(df, feature_columns)
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    if k_values is None:
        k_values = range(2, 11)

    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = model.fit_predict(X_scaled)
        inertia = model.inertia_
        silhouette = silhouette_score(X_scaled, labels) if k > 1 else np.nan
        rows.append({"k": k, "inertia": inertia, "silhouette": silhouette})
    return pd.DataFrame(rows)


def plot_elbow_silhouette(
    results: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(results["k"], results["inertia"], marker="o")
    axes[0].set_title("K-Means Elbow")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("inertia")

    axes[1].plot(results["k"], results["silhouette"], marker="o")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("silhouette")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def tune_k_q(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    k_values: Optional[Iterable[int]] = None,
    q_values: Optional[Iterable[float]] = None,
    target_range: Tuple[float, float] = (0.02, 0.08),
    random_state: int = 42,
) -> pd.DataFrame:
    if k_values is None:
        k_values = range(2, 11)
    if q_values is None:
        q_values = [90.0, 92.5, 95.0, 97.5, 99.0]

    rows = []
    for k in k_values:
        detector = KMeansAnomalyDetector(
            n_clusters=k,
            feature_columns=feature_columns,
            random_state=random_state,
        )
        detector.fit(train_df)

        for q in q_values:
            detector.set_threshold_quantile(q)
            detector.thresholds = detector._compute_thresholds(
                detector.scaler.transform(stack_features(add_features(train_df), feature_columns)[0]),
                detector.kmeans.labels_,
                q,
            )
            scored = detector.score(val_df)
            flag_rate = float(scored["anomaly_flag"].mean())
            rows.append(
                {
                    "k": k,
                    "q": q,
                    "flag_rate": flag_rate,
                    "in_target": target_range[0] <= flag_rate <= target_range[1],
                }
            )

    result = pd.DataFrame(rows)
    result["distance_from_target"] = np.abs(result["flag_rate"] - np.mean(target_range))
    result.sort_values(["in_target", "distance_from_target", "flag_rate"],ascending=[False, True, True],inplace=True)
    return result

