import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import rand_score
from pathlib import Path


'''
The code in this file implements a workflow to evaluate the stability of UMAP + HDBSCAN 
clustering when random points are removed from the dataset. This specific file is for
evaluating subclusters of the Rhythmic group. The experiment is trying to see if
similar subclusters can be achieved with less data. 
The main steps are:

1. Get ground truth labels from CSV, these are the "full_labels" that we will compare against.
2. For each value of "remove_n" values (number of points to remove), repeat the following:
    a. Randomly select which runs to remove from the dataset.
    b. Fit UMAP + HDBSCAN on the reduced dataset (after removing points).
    c. Compare the cluster labels from the reduced dataset to the full dataset labels using RAND index.
3. Aggregate results across repeats and different "remove_n" values to analyze how stability changes as more points are removed.
4. Visualize results and save to PNG


An alternative is to fit the UMAP + HDBSCAN once on the full dataset to get "full_labels", then use those 
labels as the ground truth. Theoretically, that approach should achieve higher values of RAND.
'''

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
    n_repeats=50,
    umap_params=None,
    hdbscan_params=None,
    random_seed=41,
):
    """
    Randomly remove remove_n points, recluster the reduced dataset,
    and compare reduced labels to the full-dataset labels using RAND/ARI.
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
        rand = rand_score(full_labels_subset, reduced_labels)
        n_clusters = len(set(reduced_labels)) - (1 if -1 in reduced_labels else 0)
        noise_points = np.sum(reduced_labels == -1)

        results.append({
            "remove_n": remove_n,
            "repeat": repeat,
            "n_kept": len(kept_indices),
            "ari": ari,
            "rand": rand,
            "n_clusters": n_clusters,
            "noise_points": noise_points,
        })

    return pd.DataFrame(results)


def run_ari_stability_workflow(
    X,
    full_labels,
    remove_n_values,
    n_repeats=25,
    umap_params=None,
    hdbscan_params=None,
    random_seed=41,
):
    """
    Full workflow:
    1. Cluster full dataset once.
    2. For each removal size, repeatedly remove random points.
    3. Recluster reduced dataset.
    4. Compare only shared points using RAND/ARI.
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
            mean_rand=("rand", "mean"),
            sd_rand=("rand", "std"),
            min_rand=("rand", "min"),
            max_rand=("rand", "max"),
            n_repeats=("ari", "count"),
            mean_n_clusters=("n_clusters", "mean"),
            mean_noise_points=("noise_points", "mean"),
        )
        .reset_index()
    )

    return results, summary, full_labels


# Data Loading and Preprocessing
ROOT = Path.cwd()
while not (ROOT / 'data').exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if not (ROOT / 'data').exists():
    raise FileNotFoundError("Could not locate project root containing a 'data' folder")

print(f"ROOT: {ROOT}")

# Load and merge subcluster CSVs
cluster_0 = pd.read_csv(ROOT / 'cluster_labels' /  'UMAP-HDBSCAN_Noise_Removed_subclusters' / '0_dense_rhythmic' / 'subcluster_0.csv')
cluster_0['cluster'] = 0

cluster_1 = pd.read_csv(ROOT / 'cluster_labels' /  'UMAP-HDBSCAN_Noise_Removed_subclusters' / '1_double_rhythmic' / 'subcluster_1.csv')
cluster_1['cluster'] = 1

cluster_2 = pd.read_csv(ROOT / 'cluster_labels' /  'UMAP-HDBSCAN_Noise_Removed_subclusters' / '2_single_rhythmic' / 'subcluster_2.csv')
cluster_2['cluster'] = 2

df_raw = pd.concat([cluster_0, cluster_1, cluster_2], ignore_index=True)

print(f"Loaded {len(df_raw)} rows, {len(df_raw.columns)} columns")

full_labels = df_raw["cluster"].to_numpy()

file_names = df_raw["file_name"].reset_index(drop=True)
X = df_raw.drop(columns=["file_name", "cluster", "subcluster"]).copy()

X = X.replace({"True": 1, "False": 0, True: 1, False: 0})
X = X.apply(pd.to_numeric, errors="coerce")
non_numeric = X.columns[X.isna().all()].tolist()
if non_numeric:
    print(f"Dropping non-numeric columns: {non_numeric}")
    X = X.drop(columns=non_numeric)
X = X.fillna(X.mean())

scaler    = StandardScaler()
X_scaled  = scaler.fit_transform(X)
print(f"Feature matrix: {X_scaled.shape}")

'''
We can also get the features by merging with the full dataset. 
Gets same results. 

# Use the file name and cluster labels from combined clusters for the analysis of dense, double, and single rhythmic subclusters.
cluster_labels = df_raw[["file_name", "cluster"]].copy()

non_noise_features = pd.read_csv(ROOT / 'data' / 'non_noise_features.csv')

# Extra safe: grab rows from original features that match the file names in cluster_labels
# Attach cluster labels to the true feature table
analysis_df = non_noise_features.merge(
    cluster_labels,
    on="file_name",
    how="inner"
)

# Useful sanity checks
print("Rows after merge:", len(analysis_df))
print("Cluster counts:")
print(analysis_df["cluster"].value_counts())

# Separate labels and features
full_labels = analysis_df["cluster"].to_numpy()

X = analysis_df.drop(columns=["file_name", "cluster"])

# Scale the actual features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

'''
# UMAP and HDBSCAN parameters, these produced the Rhythmic subclusters
umap_params = {
    "n_neighbors": 10,
    "min_dist": 0.001,
    "n_components": 3,
    "metric": "euclidean",
    "random_state": 41,
}

hdbscan_params = {
    "min_cluster_size": 15, # increase to merge smaller clusters into 1
    "min_samples": 7,
    "metric": "euclidean",
    "cluster_selection_epsilon": 0.03, # increase to merge nearby clusters into 1
    "cluster_selection_method": "eom",
}

# range(1, 201)
remove_n_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
N_REPEATS = 50


results, summary, full_labels = run_ari_stability_workflow(
    X=X_scaled,
    full_labels=full_labels,
    remove_n_values=remove_n_values,
    n_repeats=N_REPEATS,
    umap_params=umap_params,
    hdbscan_params=hdbscan_params,
    random_seed=41,
)

output_dir = ROOT / "Stability Testing" / "rhythmic_subclusters_stability_results"
output_dir.mkdir(parents=True, exist_ok=True)

results_path = output_dir / "rhythmic_subclusters_stability_all_results.csv"
summary_path = output_dir / "rhythmic_subclusters_stability_summary.csv"
image_path = output_dir / "rhythmic_subclusters_stability_plot.png"

results.to_csv(results_path, index=False)
summary.to_csv(summary_path, index=False)

print(summary)
print(f"Saved all results to: {results_path}")
print(f"Saved summary results to: {summary_path}")

# ── Plot: RAND and cluster count vs removal size RHYTHMIC SUBCLUSTERS ───────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

fig, ax1 = plt.subplots(figsize=(11, 5))

#summary = pd.read_csv("../data/rhythmic_subclusters_stability_results/rhythmic_subclusters_stability_summary.csv")

x     = summary["remove_n"]
y     = summary["mean_rand"]
y_std = summary["sd_rand"]

ax1.plot(x, y, color="steelblue", linewidth=2, marker="o", markersize=5,
         label="Mean RAND")
#ax1.fill_between(x, y - y_std, y + y_std, alpha=0.2, color="steelblue",
 #                label="±1 std")
ax1.set_xlabel("Points removed", fontsize=13)
ax1.set_ylabel("RAND vs full-data labels", fontsize=13, color="steelblue")
ax1.tick_params(axis="y", labelcolor="steelblue")
ax1.set_ylim(-0.05, 1.05)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax1.grid(True, alpha=0.25)

'''
Optional code to add a secondary y-axis for mean cluster count, 
and a horizontal line for the full-data cluster count.

ax2 = ax1.twinx()
ax2.plot(x, summary["mean_n_clusters"], color="tomato", linewidth=1.5,
         linestyle="--", marker="s", markersize=4, label="Mean cluster count")
ax2.axhline(len(set(full_labels)) - (1 if -1 in full_labels else 0),
            color="tomato", linewidth=1, linestyle=":", alpha=0.6,
            label=f"Full-data clusters = {len(set(full_labels)) - (1 if -1 in full_labels else 0)}")
ax2.set_ylabel("Mean cluster count", fontsize=12, color="tomato")
ax2.tick_params(axis="y", labelcolor="tomato")
ax2.set_ylim(0, max(summary["mean_n_clusters"].max() + 1, 5))
'''

lines1, labels1 = ax1.get_legend_handles_labels()
#lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1, labels1, loc="lower left", fontsize=10)

ax1.set_title(
    f"Clustering Stability Of Rhythmic Subclusters By Points Removed\n"
    f"UMAP + HDBSCAN  |  {N_REPEATS} repeats per removal size  |  "
    f"Total of {len(X_scaled)} boiling runs",
    fontsize=12,
)

plt.tight_layout()
plt.savefig(image_path, dpi=150)
plt.show()
print(f"Saved → {image_path}")
