import pandas as pd
from sklearn.neighbors import NearestNeighbors
from src.market import compute_market
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np

FEATURES = ["ret_z", "vol_z", "range_pct"]

def find_best_eps(X_scaled, min_samples=10):
    # 1. use NearestNeighbors(n_neighbors=min_samples)
    # 2. fit on X_scaled, get distances to k-th nearest neighbor
    # 3. sort distances in ascending order
    # 4. plot sorted distances — look for the "elbow"
    # 5. elbow point = good eps value

    nbrs = NearestNeighbors(n_neighbors=min_samples)
    nbrs.fit(X_scaled)
    distances, indices = nbrs.kneighbors(X_scaled)
    distances = np.sort(distances[:, -1])
    plt.plot(distances)
    plt.xlabel('Data Points')
    plt.ylabel('Distance to k-th Nearest Neighbor')
    plt.title('Elbow Method for Optimal eps')
    plt.show()

class DBSCANDetector:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
        self.scaler = None

    def fit_scaler(self, train_df):
        self.scaler = StandardScaler()
        self.scaler.fit(train_df[FEATURES])

    def score_block(self, history_df, new_block_df):
        combined = pd.concat([history_df, new_block_df]).reset_index(drop=True)
        n_history = len(history_df)  # store BEFORE dropna

        data = combined[["date", "ticker"] + FEATURES].dropna(subset=FEATURES)
        
        # count how many history rows survived dropna
        n_history_survived = (data.index < n_history).sum()

        X_scaled = self.scaler.transform(data[FEATURES])

        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = dbscan.fit_predict(X_scaled)

        # split correctly using survived count
        new_labels = labels[n_history_survived:]
        new_ids = data[data.index >= n_history][["date", "ticker"]].copy()

        new_ids["dbscan_flag"] = (new_labels == -1).astype(int)

        # merge back onto new_block_df — unscored rows get 0
        result = new_block_df.merge(
            new_ids[["date", "ticker", "dbscan_flag"]],
            on=["date", "ticker"], how="left"
        )
        result["dbscan_flag"] = result["dbscan_flag"].fillna(0).astype(int)
        
        return result

#TESTING    
# full_df = pd.read_csv("/home/p3r5eus/Documents/Stock Market Anomaly Detection/Stratex/data/processed.csv")
# full_df["date"] = pd.to_datetime(full_df["date"])
# full_df = compute_market(full_df)

# train_df = full_df[full_df["date"].dt.year == 2018].copy()
# val_df   = full_df[full_df["date"].dt.year == 2019].copy()

# # step 1 — find best eps visually
# X_train_scaled = StandardScaler().fit_transform(
#     train_df[FEATURES].dropna()
# )
# find_best_eps(X_train_scaled, min_samples=10)
# # look at the plot → pick eps at the elbow
# # step 2 — fit scaler
# detector = DBSCANDetector(eps=0.5, min_samples=10) 
# detector.fit_scaler(train_df.dropna(subset=FEATURES))

# # step 3 — score one month first to sanity check
# jan_2019 = val_df[val_df["date"].dt.month == 1].copy()

# scored = detector.score_block(train_df, jan_2019)

# print(scored["dbscan_flag"].value_counts())
# flag_rate = scored["dbscan_flag"].mean()
# print(f"Jan 2019 flag rate: {flag_rate:.2%}")  # target 2-8%
