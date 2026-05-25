from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
from market import compute_market
import matplotlib.pyplot as plt

FEATURES = ["ret_z", "vol_z", "range_pct"]

def find_best_k(X_scaled, k_range=(2, 11), save_path='silhouette_scores.png'):
    best_k = None
    best_score = -1
    
    # Lists to keep track of values for the graph
    k_values = list(range(*k_range))
    scores = []
    
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        
        # Store the score for plotting later
        scores.append(score) 
        
        if score > best_score:
            best_k = k
            best_score = score
            
    # --- Plotting the Results ---
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, scores, marker='o', linestyle='-', color='b')
    
    # Formatting the graph
    plt.title('Silhouette Score vs. Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.xticks(k_values) # Forces the x-axis to show every k value
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save a copy of the graph
    if save_path:
        # bbox_inches='tight' ensures the labels don't get cut off in the saved image
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Graph successfully saved to: {save_path}")
        
    # Display the graph in the console/notebook
    plt.show()
            
    return best_k

class KMEANSDetector:
    def __init__(self,k,q=95):
        self.k = k
        self.q = q
        self.model= None
        self.scaler = None
        self.thresholds = None
    
    def fit(self,train_df):
        X = train_df[FEATURES].values
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = KMeans(n_clusters=self.k, random_state=42)
        
        labels = self.model.fit_predict(X_scaled)
        centers = self.model.cluster_centers_

        self.thresholds = {}
        for c in range(self.k):
            points_in_c = X_scaled[labels == c]   # already scaled
            distances = np.linalg.norm(points_in_c - centers[c], axis=1)
            self.thresholds[c] = np.percentile(distances, self.q)

    def score(self, df):
        data = df[["date","ticker"]+FEATURES].dropna(subset=FEATURES)
        X_scaled = self.scaler.transform(data[FEATURES].values)
        
        labels = self.model.predict(X_scaled)
        centers = self.model.cluster_centers_
        
        # compute distance to assigned centroid for each point
        distances = np.linalg.norm(X_scaled - centers[labels], axis=1)
        
        # flag if distance exceeds that cluster's threshold
        flags = np.array([
            1 if distances[i] > self.thresholds[labels[i]] else 0
            for i in range(len(labels))
        ])
        
        result = data.copy()
        result["kmeans_flag"] = flags
        result["kmeans_distance"] = distances
        
        df = df.merge(result[["date", "ticker", "kmeans_flag", "kmeans_distance"]],
                    on=["date", "ticker"], how="left")
        df["kmeans_flag"] = df["kmeans_flag"].fillna(0).astype(int)
        
        return df

# full_df = pd.read_csv("/home/p3r5eus/Documents/Stock Market Anomaly Detection/Stratex/data/processed.csv")
# full_df["date"] = pd.to_datetime(full_df["date"])
# full_df = compute_market(full_df)
# train_df = full_df[full_df["date"].dt.year == 2018].copy()
# val_df   = full_df[full_df["date"].dt.year == 2019].copy()

# best_k = find_best_k(  # pass scaled train features
#     StandardScaler().fit_transform(train_df[FEATURES].dropna())
# )
# print("Best k:", best_k)

# detector = KMEANSDetector(k=best_k, q=95)
# detector.fit(train_df.dropna(subset=FEATURES))

# val_scored = detector.score(val_df)
# flag_rate = val_scored["kmeans_flag"].mean()
# print(f"Val flag rate: {flag_rate:.2%}")  # target: 2–8%