import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.metrics import adjusted_rand_score
from pathlib import Path

def fit_umap_hdbscan(X, umap_params=None, hdbscan_params=None):
    """
    Fit UMAP + HDBSCAN and return cluster labels.
    X should be a pandas DataFrame or numpy array.
    """
    umap_params = umap_params or {}
    hdbscan_params = hdbscan_params or {}

    reducer = UMAP(**umap_params)
    X_umap = reducer.fit_transform(X)

    clusterer = HDBSCAN(**hdbscan_params)
    labels = clusterer.fit_predict(X_umap)

    return labels


def ari_after_removing_points(
    X,
    full_labels,
    remove_n,
    n_repeats=10,
    umap_params=None,
    hdbscan_params=None,
    random_seed=42,
):
    """
    Randomly remove remove_n points, recluster the reduced dataset,
    and compare reduced labels to the full-dataset labels using ARI.
    """
    rng = np.random.default_rng(random_seed)

    n = len(X)
    all_indices = np.arange(n)

    results = []

    for repeat in range(n_repeats):
        removed_indices = rng.choice(all_indices, size=remove_n, replace=False)
        kept_indices = np.setdiff1d(all_indices, removed_indices)

        X_reduced = X.iloc[kept_indices] if isinstance(X, pd.DataFrame) else X[kept_indices]

        reduced_umap_params = dict(umap_params or {})
        reduced_umap_params["random_state"] = random_seed + repeat

        reduced_labels = fit_umap_hdbscan(
            X_reduced,
            umap_params=reduced_umap_params,
            hdbscan_params=hdbscan_params,
        )

        full_labels_subset = full_labels[kept_indices]

        ari = adjusted_rand_score(full_labels_subset, reduced_labels)

        results.append({
            "remove_n": remove_n,
            "repeat": repeat,
            "n_kept": len(kept_indices),
            "ari": ari,
        })

    return pd.DataFrame(results)


def run_ari_stability_workflow(
    X,
    full_labels,
    remove_n_values,
    n_repeats=10,
    umap_params=None,
    hdbscan_params=None,
    random_seed=42,
):
    """
    Full workflow:
    1. Cluster full dataset once.
    2. For each removal size, repeatedly remove random points.
    3. Recluster reduced dataset.
    4. Compare only shared points using ARI.
    """

    all_results = []

    for remove_n in remove_n_values:
        df = ari_after_removing_points(
            X=X,
            full_labels=full_labels,
            remove_n=remove_n,
            n_repeats=n_repeats,
            umap_params=umap_params,
            hdbscan_params=hdbscan_params,
            random_seed=random_seed + remove_n,
        )
        all_results.append(df)

    results = pd.concat(all_results, ignore_index=True)

    summary = (
        results
        .groupby("remove_n")
        .agg(
            mean_ari=("ari", "mean"),
            sd_ari=("ari", "std"),
            min_ari=("ari", "min"),
            max_ari=("ari", "max"),
            n_repeats=("ari", "count"),
        )
        .reset_index()
    )

    return results, summary, full_labels


#  example usage

ROOT = Path.cwd()
while not (ROOT / 'data').exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if not (ROOT / 'data').exists():
    raise FileNotFoundError("Could not locate project root containing a 'data' folder")

print(f"ROOT: {ROOT}")


#FEATURES_CSV = "../data/non_noise_features.csv"
# can use non_noise_features.csv for reduced clustering, it's assumed combined_clusters
# has the same data points

# Load and merge cluster CSVs
cluster_0 = pd.read_csv(ROOT / 'data' / '3d_umap_clusters' / 'cluster_0.csv')
cluster_0['cluster'] = 0

cluster_1 = pd.read_csv(ROOT / 'data' / '3d_umap_clusters' / 'cluster_1.csv')
cluster_1['cluster'] = 1

cluster_2 = pd.read_csv(ROOT / 'data' / '3d_umap_clusters' / 'cluster_2.csv')
cluster_2['cluster'] = 2

# Combine all clusters into one DataFrame
combined_clusters = pd.concat([cluster_0, cluster_1, cluster_2], ignore_index=True)

# Save to a new CSV
#combined_clusters.to_csv(ROOT / 'data' / '3d_umap_clusters' / 'combined_clusters.csv', index=False)

#print("Combined clusters saved to ../data/3d_umap_clusters/combined_clusters.csv")

# Use the combined clusters for the analysis
full_labels = combined_clusters['cluster'].values
X = combined_clusters.drop(columns = ['file_name', 'cluster'])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

umap_params = {
    "n_neighbors": 15,
    "min_dist": 0.1,
    "n_components": 2,
    "metric": "euclidean",
}

hdbscan_params = {
    "min_cluster_size": 20, # increase to merge smaller clusters into 1
    "min_samples": 10,
    "metric": "euclidean",
}

remove_n_values = [50, 100, 150]

results, summary, full_labels = run_ari_stability_workflow(
    X=X_scaled,
    full_labels=combined_clusters['cluster'],
    remove_n_values=remove_n_values,
    n_repeats=10,
    umap_params=umap_params,
    hdbscan_params=hdbscan_params,
    random_seed=123,
)

print(summary)