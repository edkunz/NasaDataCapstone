"""
Generate umap_2d.json and umap_3d.json for the App.
Run from the project root:  python generate_umap_json.py
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan

ROOT = os.path.dirname(os.path.abspath(__file__))
FEATURES_CSV = os.path.join(ROOT, "data", "features_final.csv")
OUT_DIR = os.path.join(ROOT, "App", "public", "data")
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading features...")
df = pd.read_csv(FEATURES_CSV)
file_names = df["file_name"].tolist()
X = df.drop(columns=["file_name"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

UMAP_PARAMS = dict(n_neighbors=20, min_dist=0.05, metric="euclidean", random_state=42)
HDBSCAN_PARAMS = dict(min_cluster_size=20, min_samples=10, metric="euclidean")

def run_and_save(n_components: int):
    tag = f"{n_components}d"
    print(f"\nFitting UMAP {n_components}D...")
    reducer = umap.UMAP(n_components=n_components, **UMAP_PARAMS)
    X_umap = reducer.fit_transform(X_scaled)

    print(f"Fitting HDBSCAN for {n_components}D...")
    clusterer = hdbscan.HDBSCAN(**HDBSCAN_PARAMS)
    labels = clusterer.fit_predict(X_umap)

    points = []
    for i, fn in enumerate(file_names):
        p = {
            "file_name": fn,
            "cluster": int(labels[i]),
            "umap1": float(X_umap[i, 0]),
            "umap2": float(X_umap[i, 1]),
        }
        if n_components == 3:
            p["umap3"] = float(X_umap[i, 2])
        points.append(p)

    payload = {"dims": n_components, "points": points}
    out_path = os.path.join(OUT_DIR, f"umap_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"Wrote {len(points)} points -> {out_path}")
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  Clusters: {n_clusters}  Noise points: {int((labels == -1).sum())}")

run_and_save(2)
run_and_save(3)
print("\nDone!")
