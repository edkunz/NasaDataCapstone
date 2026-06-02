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
        #reduced_umap_params["random_state"] = random_seed + repeat

        reduced_labels = fit_umap_hdbscan(
            X_reduced,
            umap_params=reduced_umap_params,
            hdbscan_params=hdbscan_params,
        )

        full_labels_subset = full_labels[kept_indices]

        ari = adjusted_rand_score(full_labels_subset, reduced_labels)
        n_clusters = len(set(reduced_labels)) - (1 if -1 in reduced_labels else 0)
        noise_points = np.sum(reduced_labels == -1)

        results.append({
            "remove_n": remove_n,
            "repeat": repeat,
            "n_kept": len(kept_indices),
            "ari": ari,
            "n_clusters": n_clusters,
            "noise_points": noise_points,
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
            mean_n_clusters=("n_clusters", "mean"),
            mean_noise_points=("noise_points", "mean"),
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
cluster_0 = pd.read_csv(ROOT / 'visuals' /  'rep_samples' / 'UMAP-HDBSCAN_Noise_Removed' / '0' / 'cluster_0.csv')
cluster_0['cluster'] = 0

cluster_1 = pd.read_csv(ROOT / 'visuals' / 'rep_samples' /  'UMAP-HDBSCAN_Noise_Removed' / '1' / 'cluster_1.csv')
cluster_1['cluster'] = 1

cluster_2 = pd.read_csv(ROOT / 'visuals' /  'rep_samples' / 'UMAP-HDBSCAN_Noise_Removed' / '2' /'cluster_2.csv')
cluster_2['cluster'] = 2

# Combine all clusters into one DataFrame
combined_clusters = pd.concat([cluster_0, cluster_1, cluster_2], ignore_index=True)

# Save to a new CSV
combined_clusters.to_csv(ROOT / 'visuals' / 'rep_samples' / 'UMAP-HDBSCAN_Noise_Removed' / 'combined_clusters.csv', index=False)

print("Combined clusters saved to ../visuals/rep_samples/UMAP-HDBSCAN_Noise_Removed/combined_clusters.csv")

# Use the combined clusters for the analysis
full_labels = combined_clusters['cluster'].values
X = combined_clusters.drop(columns = ['file_name', 'cluster'])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

non_noise_features = pd.read_csv(ROOT / 'data' / 'non_noise_features.csv')
non_noise_features_scaled = scaler.transform(non_noise_features.drop(columns=['file_name']))

umap_params = {
    "n_neighbors": 10,
    "min_dist": 0.001,
    "n_components": 3,
    "metric": "euclidean",
    "random_state": 41,
}

hdbscan_params = {
    "min_cluster_size": 40, # increase to merge smaller clusters into 1
    "min_samples": 5,
    "metric": "euclidean",
}

#fit initial clustering, compare to this: 
reducer = UMAP(**umap_params)
X_umap = reducer.fit_transform(non_noise_features_scaled)

clusterer = HDBSCAN(**hdbscan_params)
full_labels = clusterer.fit_predict(X_umap)

# range(1, 201)
remove_n_values = [10, 50, 100]

results, summary, full_labels = run_ari_stability_workflow(
    X=non_noise_features_scaled,
    full_labels=full_labels,
    remove_n_values=remove_n_values,
    n_repeats=15,
    umap_params=umap_params,
    hdbscan_params=hdbscan_params,
    random_seed=123,
)

output_dir = ROOT / "data" / "ari_stability_results"
output_dir.mkdir(parents=True, exist_ok=True)

results_path = output_dir / "ari_stability_all_results.csv"
summary_path = output_dir / "ari_stability_summary.csv"

results.to_csv(results_path, index=False)
summary.to_csv(summary_path, index=False)

print(summary)
print(f"Saved all results to: {results_path}")
print(f"Saved summary results to: {summary_path}")
