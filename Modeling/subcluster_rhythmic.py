from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from umap.umap_ import UMAP
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


def save_boilings_plot(embedding, boilings, output_dir):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=boilings, cmap="viridis", s=30)
    plt.title("Rhythmic Subclusters Colored by Number of Boilings")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Number of Boilings")
    plt.tight_layout()
    plt.savefig(output_dir / "rhythmic_subclusters_boilings_plot.png")
    plt.close()


def save_hand_labeled_plot(embedding, labels, output_dir):
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="Set2", s=30)
    plt.title("Rhythmic Subclusters with Hand Labels")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.colorbar(label="Hand Label")
    plt.tight_layout()
    plt.savefig(output_dir / "rhythmic_subclusters_hand_labels_plot.png")
    plt.close()


def find_plot_for_name(run_name, boiling_plot_dir):
    if boiling_plot_dir is None or pd.isna(run_name):
        return None

    run_name = str(run_name).strip()
    stem = Path(run_name).stem

    exact = boiling_plot_dir / f"{stem}.png"
    if exact.exists():
        return exact

    for f in boiling_plot_dir.glob("*.png"):
        fstem = f.stem.strip()
        if fstem == stem or fstem.startswith(stem) or stem in fstem:
            return f

    return None


def main():
    project_root = Path(__file__).resolve().parent.parent

    # input files, with code to combine more than one csv if needed
    cluster_2_path = project_root / "visuals" / "rep_samples" / "UMAP-HDBSCAN_Noise_Removed" / "2" / "cluster_2.csv"
    #cluster_2_path = project_root / "visuals" / "rep_samples" / "UMAP-HDBSCAN" / "2" / "cluster_2.csv"
    features_path = project_root / "data" / "features.csv"

    # output folder
    output_root = project_root / "data" / "rhythmic_subcluster_final"
    output_root.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    print(f"Using inputs:\n - {cluster_2_path}")

    #df = pd.read_csv(cluster_1_path)
    #df2 = pd.read_csv(cluster_2_path)
    #df = pd.concat([df1, df2], ignore_index=True)
    df_cluster_2 = pd.read_csv(cluster_2_path)
    df_features = pd.read_csv(features_path)

    if "file_name" not in df_cluster_2.columns:
        raise ValueError("Expected column 'file_name' in cluster_2.csv")

    if "file_name" not in df_features.columns:
        raise ValueError("Expected column 'file_name' in features.csv")

    selected_file_names = df_cluster_2["file_name"].astype(str).tolist()

    df_features["file_name"] = df_features["file_name"].astype(str)

    df = df_features[df_features["file_name"].isin(selected_file_names)].copy()

    # preserve same order as cluster_2.csv
    order_map = {fn: i for i, fn in enumerate(selected_file_names)}
    df["__order"] = df["file_name"].map(order_map)
    df = df.sort_values("__order").drop(columns="__order").reset_index(drop=True)

    print(f"Combined rows: {len(df)}")

    if "file_name" not in df.columns:
        raise ValueError("Expected column 'file_name' in combined cluster_1.csv and cluster_2.csv")

    # Separate metadata and features
    file_names = df["file_name"].copy()
    X = df.drop(columns=["file_name"]).copy()

    # keep only numeric columns
    X = X.select_dtypes(include=[np.number]).copy()

    # drop NaNs
    keep_mask = ~X.isna().any(axis=1)
    dropped = int((~keep_mask).sum())
    if dropped > 0:
        print(f"Dropping {dropped} rows due to NaN values.")

    X = X.loc[keep_mask].copy()
    df = df.loc[keep_mask].copy()
    file_names = file_names.loc[keep_mask].copy()

    if X.empty:
        raise ValueError("No usable numeric feature rows remain after cleaning.")

    print(f"Feature matrix shape after cleaning: {X.shape}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit UMAP (your current subcluster tuning)
    umap_model = UMAP(
        n_neighbors=17,
        min_dist=0.0,
        n_components=2,
        metric="euclidean",
        random_state=42
    )
    embedding = umap_model.fit_transform(X_scaled)

    # optional hand-labeled plot if labels exist
    labels_xlsx = project_root / "data" / "Labeling.xlsx"
    if labels_xlsx.exists():
        try:
            df_labels = pd.read_excel(labels_xlsx)
            df_labels = df_labels.dropna(subset=["Label"]).copy()
            df_labels["Label"] = df_labels["Label"].astype(int)
            df_labels["match_name"] = "MATLAB " + df_labels["File"].astype(str)

            df["match_name"] = file_names.astype(str)
            df = df.merge(
                df_labels[["match_name", "Label"]],
                on="match_name",
                how="left"
            )

            valid_labels_mask = ~df["Label"].isna()
            if valid_labels_mask.sum() > 0:
                hand_labels = df.loc[valid_labels_mask, "Label"].values
                emb_plot = embedding[valid_labels_mask.values]
                save_hand_labeled_plot(emb_plot, hand_labels, output_root)
                print("Hand-labeled plot saved!")
        except Exception as e:
            print(f"Hand-labeled plot skipped: {e}")

    # Fit HDBSCAN (your current subcluster tuning)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=15,
        min_samples=12,
        metric="euclidean",
        cluster_selection_epsilon=0.0,
        cluster_selection_method="eom"
    )
    sub_labels = clusterer.fit_predict(embedding)

    # attach subcluster labels
    df["subcluster"] = sub_labels
    df["file_name"] = file_names.values

    # save full output
    df.to_csv(output_root / "rhythmic_subcluster_output.csv", index=False)

    # counts
    counts = (
        df["subcluster"]
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
            [X.reset_index(drop=True), df[["subcluster"]].reset_index(drop=True)],
            axis=1
        )
        .groupby("subcluster")
        .mean(numeric_only=True)
    )
    feature_means.to_csv(output_root / "rhythmic_subcluster_feature_means.csv")

    # silhouette
    if len(set(sub_labels)) > 1:
        sil = silhouette_score(embedding, sub_labels)
        print(f"Silhouette score: {sil:.3f}")

    # evaluation prints
    print("\n--- Evaluation ---")
    n_clusters = len(set(sub_labels)) - (1 if -1 in sub_labels else 0)
    print(f"Number of clusters (excluding noise): {n_clusters}")

    n_noise = np.sum(sub_labels == -1)
    print(f"Number of noise points: {n_noise}")

    if hasattr(clusterer, "cluster_persistence_"):
        print("\nCluster persistence:")
        for i, p in enumerate(clusterer.cluster_persistence_):
            print(f"Cluster {i}: {p:.3f}")

    # plots
    if embedding.shape[1] >= 2:
        save_plots(embedding[:, :2], sub_labels, output_root)

    # boilings plot if available
    boilings_col = None
    for c in ["a0_num_boilings_y", "a0_num_boilings", "a0_num_boilings_x"]:
        if c in df.columns:
            boilings_col = c
            break

    if boilings_col is not None and embedding.shape[1] >= 2:
        boilings_vals = df[boilings_col].values
        save_boilings_plot(embedding[:, :2], boilings_vals, output_root)
        print("Boilings plot saved!")

    # locate boiling plots
    possible_plot_dirs = [
        project_root / "visuals" / "boiling_plots",
        project_root / "data" / "boiling_plots",
    ]
    boiling_plot_dir = None
    for p in possible_plot_dirs:
        if p.exists():
            boiling_plot_dir = p
            break

    # save one folder per subcluster with csv + images
    for sub_id in sorted(df["subcluster"].dropna().unique()):
        sub_dir = output_root / f"subcluster_{sub_id}"
        sub_dir.mkdir(parents=True, exist_ok=True)

        sub_df = df[df["subcluster"] == sub_id].copy()
        sub_df.to_csv(sub_dir / f"subcluster_{sub_id}.csv", index=False)

        for fn in sub_df["file_name"]:
            img_path = find_plot_for_name(fn, boiling_plot_dir)
            if img_path is not None and img_path.exists():
                shutil.copy2(img_path, sub_dir / img_path.name)

    print("\nDone. Outputs saved to:")
    print(output_root)


if __name__ == "__main__":
    main()