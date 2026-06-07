from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from umap import UMAP


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURES_CSV = PROJECT_ROOT / "data" / "non_noise_features.csv"

# existing subcluster output that already has -1 labels
SUBCLUSTER_CSV = PROJECT_ROOT / "data" / "rhythmic_subcluster_final" / "rhythmic_subcluster_output.csv"

OUT_DIR = PROJECT_ROOT / "data" / "rhythmic_subcluster_final" / "noise_assignment_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# best tuning params from your hyperparameter tuning
UMAP_PARAMS = dict(
    n_neighbors=10,
    min_dist=0.05,
    n_components=2,
    metric="euclidean",
    random_state=42,
)


def main():
    print("Loading data...")

    df_sub = pd.read_csv(SUBCLUSTER_CSV)
    df_feat = pd.read_csv(FEATURES_CSV)

    if "file_name" not in df_sub.columns:
        raise ValueError("rhythmic_subcluster_output.csv needs a file_name column")

    if "file_name" not in df_feat.columns:
        raise ValueError("non_noise_features.csv needs a file_name column")

    if "subcluster" not in df_sub.columns:
        raise ValueError("rhythmic_subcluster_output.csv needs a subcluster column")

    df_sub["file_name"] = df_sub["file_name"].astype(str)
    df_feat["file_name"] = df_feat["file_name"].astype(str)

    # merge labels with features
    df = df_sub[["file_name", "subcluster"]].merge(
        df_feat,
        on="file_name",
        how="inner"
    )

    print(f"Rows merged: {len(df)}")

    feature_cols = [c for c in df.columns if c not in ["file_name", "subcluster"]]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # recreate UMAP embedding for plotting using best tuning
    print("Creating UMAP plot coordinates...")
    umap_model = UMAP(**UMAP_PARAMS)
    embedding = umap_model.fit_transform(X_scaled)

    labels = df["subcluster"].values.copy()

    real_clusters = sorted([c for c in np.unique(labels) if c != -1])
    noise_idx = np.where(labels == -1)[0]

    print("\nOriginal subcluster counts:")
    print(df["subcluster"].value_counts().sort_index())

    print(f"\nNoise points to assign: {len(noise_idx)}")

    if len(real_clusters) == 0:
        raise ValueError("No real clusters found. Cannot assign noise.")

    # centroid of each real subcluster in scaled feature space
    centroids = []
    for c in real_clusters:
        centroids.append(X_scaled[labels == c].mean(axis=0))
    centroids = np.vstack(centroids)

    forced_labels = labels.copy()
    assignment_rows = []

    for idx in noise_idx:
        x = X_scaled[idx].reshape(1, -1)

        cosine_scores = cosine_similarity(x, centroids).flatten()

        pearson_scores = []
        for centroid in centroids:
            corr = np.corrcoef(X_scaled[idx], centroid)[0, 1]
            pearson_scores.append(corr)

        pearson_scores = np.array(pearson_scores)

        # assign to highest Pearson correlation
        best_pos = int(np.nanargmax(pearson_scores))
        assigned_cluster = real_clusters[best_pos]

        forced_labels[idx] = assigned_cluster

        assignment_rows.append({
            "file_name": df.loc[idx, "file_name"],
            "original_subcluster": -1,
            "forced_subcluster": assigned_cluster,
            "best_pearson_corr_to_cluster_centroid": pearson_scores[best_pos],
            "best_cosine_similarity_to_cluster_centroid": cosine_scores[best_pos],
            "umap_1": embedding[idx, 0],
            "umap_2": embedding[idx, 1],
        })

    df["subcluster_forced"] = forced_labels
    df["umap_1"] = embedding[:, 0]
    df["umap_2"] = embedding[:, 1]

    assignments = pd.DataFrame(assignment_rows)

    # save forced no-noise output
    df.to_csv(
        OUT_DIR / "rhythmic_subcluster_output_forced_no_noise.csv",
        index=False
    )

    assignments.to_csv(
        OUT_DIR / "noise_to_nearest_subcluster_assignments.csv",
        index=False
    )

    forced_counts = (
        pd.Series(forced_labels)
        .value_counts()
        .sort_index()
        .rename_axis("subcluster")
        .reset_index(name="count")
    )

    forced_counts.to_csv(
        OUT_DIR / "forced_subcluster_counts.csv",
        index=False
    )

    # save plot in same style as earlier subcluster plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=forced_labels,
        cmap="tab10",
        s=30
    )
    plt.title("Rhythmic Subclusters Forced No Noise")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.colorbar(scatter, label="Forced Subcluster")
    plt.tight_layout()
    plt.savefig(
        OUT_DIR / "rhythmic_subclusters_forced_no_noise_plot.png",
        dpi=160
    )
    plt.close()

    print("\nForced subcluster counts:")
    print(forced_counts)

    print("\nSaved outputs to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()