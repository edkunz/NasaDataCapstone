from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import umap
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def save_plots(embedding, labels, output_dir):
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10", s=30)
    plt.title("Rhythmic Subclusters (UMAP)")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.colorbar(label="Subcluster")
    plt.tight_layout()
    plt.savefig(output_dir / "rhythmic_subclusters_plot.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-csv", required=True)
    parser.add_argument("--feature-csv", required=True)
    parser.add_argument("--target-cluster", type=int, required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    cluster_csv = Path(args.cluster_csv)
    feature_csv = Path(args.feature_csv)

    output_root = project_root / "data" / "rhythmic_subclusters"
    output_root.mkdir(parents=True, exist_ok=True)

    print("Loading data...")

    df_cluster = pd.read_csv(cluster_csv)
    df_features = pd.read_csv(feature_csv)

    # --- merge ---
    if "name" in df_cluster.columns:
        name_col = "name"
    else:
        name_col = df_cluster.columns[0]

    if "name" in df_features.columns:
        feat_name_col = "name"
    else:
        feat_name_col = df_features.columns[0]

    df_cluster["match_name"] = df_cluster[name_col].astype(str)
    df_features["match_name"] = df_features[feat_name_col].astype(str)

    df_merged = df_cluster.merge(df_features, on="match_name", how="inner")

    if df_merged.empty:
        raise ValueError("Merge failed: no matching rows")

    print(f"Merged rows: {len(df_merged)}")

    # --- filter target cluster ---
    df_target = df_merged[df_merged["cluster"] == args.target_cluster].copy()

    if df_target.empty:
        raise ValueError(f"No rows found for cluster {args.target_cluster}")

    print(f"Rows in target cluster: {len(df_target)}")

    # --- extract features ---
    numeric_cols = df_target.select_dtypes(include=[np.number]).columns.tolist()

    # remove cluster labels
    drop_cols = ["cluster"]
    numeric_cols = [c for c in numeric_cols if c not in drop_cols]

    X = df_target[numeric_cols].copy()

    # drop NaNs
    X = X.dropna()

    print(f"Feature matrix shape: {X.shape}")

    # --- scale ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- UMAP ---
    reducer = umap.UMAP(
        n_neighbors=8,
        min_dist=0.01,
        n_components=2,
        metric="euclidean",
        random_state=42
    )
    embedding = reducer.fit_transform(X_scaled)

    # --- HDBSCAN ---
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=6,
        min_samples=3,
        prediction_data=True
    )
    sub_labels = clusterer.fit_predict(embedding)

    # --- save outputs ---
    df_target = df_target.loc[X.index].copy()
    df_target["subcluster"] = sub_labels

    df_target.to_csv(output_root / "rhythmic_subcluster_output.csv", index=False)

    # counts
    counts = (
        df_target["subcluster"]
        .value_counts()
        .sort_index()
        .rename_axis("subcluster")
        .reset_index(name="count")
    )
    counts.to_csv(output_root / "rhythmic_subcluster_counts.csv", index=False)

    print("Subcluster counts:")
    print(counts)

    # feature means
    feature_means = (
        pd.concat(
            [X.reset_index(drop=True), df_target[["subcluster"]].reset_index(drop=True)],
            axis=1
        )
        .groupby("subcluster")
        .mean(numeric_only=True)
    )
    feature_means.to_csv(output_root / "rhythmic_subcluster_feature_means.csv")

    # silhouette (only if >1 cluster)
    if len(set(sub_labels)) > 1:
        sil = silhouette_score(embedding, sub_labels)
        print(f"Silhouette score: {sil:.3f}")

    # plot
    save_plots(embedding, sub_labels, output_root)

    print("\nDone. Outputs saved to:")
    print(output_root)


if __name__ == "__main__":
    main()