from pathlib import Path
import shutil
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def find_column(df, candidates, required=False):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise ValueError(f"Could not find any of these columns: {candidates}")
    return None


def guess_cluster_column(df):
    candidates = ["cluster", "group", "label", "cluster_label", "hdbscan_cluster", "final_cluster"]
    col = find_column(df, candidates, required=False)
    if col:
        return col
    for c in df.columns:
        lc = c.lower()
        if "cluster" in lc or "group" in lc:
            return c
    raise ValueError("Could not detect main cluster column.")


def guess_name_column(df):
    candidates = ["name", "run_name", "run", "filename", "file_name", "png_name", "plot_name"]
    col = find_column(df, candidates, required=False)
    if col:
        return col
    object_cols = [c for c in df.columns if df[c].dtype == object]
    if object_cols:
        return object_cols[0]
    raise ValueError("Could not detect run name column.")


def clean_run_name(x):
    if pd.isna(x):
        return ""

    s = str(x).strip()

    # remove quotes
    s = s.replace('"', "").replace("'", "")

    # remove extension
    if s.lower().endswith(".csv"):
        s = s[:-4]
    if s.lower().endswith(".png"):
        s = s[:-4]

    # fix weird spacing before punctuation/extensions:
    # "Run8 .csv" -> "Run8"
    # "Run8 .png" -> "Run8"
    s = s.replace(" .", ".")

    # collapse repeated spaces
    s = " ".join(s.split())

    return s.strip()


def build_feature_matrix(df, exclude_cols):
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    drop_cols = set()
    for c in numeric_df.columns:
        lc = c.lower()
        if c in exclude_cols:
            drop_cols.add(c)
        elif "cluster" in lc or "group" in lc or "label" in lc:
            drop_cols.add(c)
        elif "umap" in lc:
            drop_cols.add(c)
        elif "subcluster" in lc:
            drop_cols.add(c)
        elif "prob" in lc or "score" in lc or "membership" in lc:
            drop_cols.add(c)

    X = numeric_df.drop(columns=list(drop_cols), errors="ignore")

    if X.shape[1] == 0:
        raise ValueError("No usable numeric feature columns were found in the feature CSV.")

    return X


def clear_old_subcluster_folders(output_root):
    for path in output_root.iterdir():
        if path.is_dir() and path.name.startswith("subcluster_"):
            shutil.rmtree(path)


def exact_png_lookup(boiling_plots_dir):
    pngs = list(boiling_plots_dir.glob("*.png"))

    by_name = {}
    by_stem = {}
    by_clean_name = {}
    by_clean_stem = {}

    for p in pngs:
        by_name[p.name] = p
        by_stem[p.stem] = p
        by_clean_name[clean_run_name(p.name)] = p
        by_clean_stem[clean_run_name(p.stem)] = p

    return pngs, by_name, by_stem, by_clean_name, by_clean_stem


def copy_matching_pngs(df_sub, name_col, subcluster_col, boiling_plots_dir, output_root):
    pngs, by_name, by_stem, by_clean_name, by_clean_stem = exact_png_lookup(boiling_plots_dir)
    missing = []
    copied = 0
    debug_rows = []

    for _, row in df_sub.iterrows():
        raw_name = str(row[name_col]).strip()
        clean_name = clean_run_name(raw_name)
        subcluster = row[subcluster_col]

        cluster_folder = output_root / f"subcluster_{subcluster}"
        cluster_folder.mkdir(parents=True, exist_ok=True)

        match = None
        match_type = None

        # exact filename
        if raw_name in by_name:
            match = by_name[raw_name]
            match_type = "exact_filename"

        # exact stem
        elif raw_name in by_stem:
            match = by_stem[raw_name]
            match_type = "exact_stem"

        # add .png to raw name
        elif f"{raw_name}.png" in by_name:
            match = by_name[f"{raw_name}.png"]
            match_type = "raw_plus_png"

        # cleaned filename
        elif clean_name in by_clean_name:
            match = by_clean_name[clean_name]
            match_type = "clean_filename"

        # cleaned stem
        elif clean_name in by_clean_stem:
            match = by_clean_stem[clean_name]
            match_type = "clean_stem"

        else:
            for p in pngs:
                if clean_run_name(p.name) == clean_name or clean_run_name(p.stem) == clean_name:
                    match = p
                    match_type = "fallback_clean_match"
                    break

        if match is None:
            missing.append(raw_name)
            debug_rows.append({
                "csv_name": raw_name,
                "csv_name_clean": clean_name,
                "matched_png": "",
                "match_type": "missing"
            })
            continue

        shutil.copy2(match, cluster_folder / match.name)
        copied += 1
        debug_rows.append({
            "csv_name": raw_name,
            "csv_name_clean": clean_name,
            "matched_png": match.name,
            "match_type": match_type
        })

    debug_df = pd.DataFrame(debug_rows)
    debug_df.to_csv(output_root / "png_match_debug.csv", index=False)

    return missing, copied


def save_plots(embedding, labels, out_dir):
    plt.figure(figsize=(9, 7))
    sc = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        cmap="tab10",
        s=24
    )
    plt.xlabel("Local UMAP 1")
    plt.ylabel("Local UMAP 2")
    plt.title("Subclustering of Rhythmic Cluster")
    plt.colorbar(sc, label="Subcluster")
    plt.tight_layout()
    plt.savefig(out_dir / "rhythmic_subcluster_umap.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 7))
    colors = np.where(labels == -1, "lightgray", "black")
    plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=24)
    plt.xlabel("Local UMAP 1")
    plt.ylabel("Local UMAP 2")
    plt.title("Noise Check (-1 in gray)")
    plt.tight_layout()
    plt.savefig(out_dir / "rhythmic_subcluster_noise_check.png", dpi=200, bbox_inches="tight")
    plt.close()


def compute_scores(embedding, labels):
    mask = labels != -1
    unique_clusters = np.unique(labels[mask])

    results = {
        "n_total_points": len(labels),
        "n_noise": int(np.sum(labels == -1)),
        "n_clustered_points": int(np.sum(mask)),
        "n_clusters_excluding_noise": int(len(unique_clusters)),
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,
    }

    if len(unique_clusters) >= 2 and np.sum(mask) >= 3:
        emb_eval = embedding[mask]
        lab_eval = labels[mask]
        results["silhouette"] = float(silhouette_score(emb_eval, lab_eval))
        results["davies_bouldin"] = float(davies_bouldin_score(emb_eval, lab_eval))
        results["calinski_harabasz"] = float(calinski_harabasz_score(emb_eval, lab_eval))

    return results


def main():
    parser = argparse.ArgumentParser(description="Subcluster the rhythmic cluster using features_no_time_corr.csv")
    parser.add_argument("--cluster-csv", default="umap_hdbscan_cluster_output.csv")
    parser.add_argument("--feature-csv", default="data/features_no_time_corr.csv")
    parser.add_argument("--target-cluster", type=int, default=2)

    parser.add_argument("--n-neighbors", type=int, default=8)
    parser.add_argument("--min-dist", type=float, default=0.01)
    parser.add_argument("--n-components", type=int, default=2)
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument("--min-cluster-size", type=int, default=6)
    parser.add_argument("--min-samples", type=int, default=3)

    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    modeling_dir = script_path.parent
    project_root = modeling_dir.parent

    cluster_csv_path = modeling_dir / args.cluster_csv
    feature_csv_path = project_root / args.feature_csv
    boiling_plots_dir = project_root / "visuals" / "boiling_plots_fixed"
    output_root = project_root / "data" / "rhythmic_subclusters"
    output_root.mkdir(parents=True, exist_ok=True)

    clear_old_subcluster_folders(output_root)

    print(f"Reading cluster CSV: {cluster_csv_path}")
    df_cluster = pd.read_csv(cluster_csv_path)

    print(f"Reading feature CSV: {feature_csv_path}")
    df_feat = pd.read_csv(feature_csv_path)

    cluster_col = guess_cluster_column(df_cluster)
    cluster_name_col = guess_name_column(df_cluster)
    feat_name_col = guess_name_column(df_feat)

    print(f"Detected main cluster column: {cluster_col}")
    print(f"Detected cluster-name column: {cluster_name_col}")
    print(f"Detected feature-name column: {feat_name_col}")
    print(f"Using PNG folder: {boiling_plots_dir}")

    df_cluster["match_name"] = df_cluster[cluster_name_col].apply(clean_run_name)
    df_feat["match_name"] = df_feat[feat_name_col].apply(clean_run_name)

    df_target = df_cluster[df_cluster[cluster_col] == args.target_cluster].copy()
    if df_target.empty:
        raise ValueError(f"No rows found for target cluster {args.target_cluster}.")

    print(f"Rows in target cluster {args.target_cluster}: {len(df_target)}")

    df_merged = df_target.merge(
        df_feat,
        on="match_name",
        how="inner",
        suffixes=("_cluster", "_feat")
    )

    if df_merged.empty:
        raise ValueError("Merge found 0 matching rows between cluster CSV and feature CSV.")

    print(f"Rows matched with features: {len(df_merged)}")

    if f"{cluster_name_col}_cluster" in df_merged.columns:
        final_name_col = f"{cluster_name_col}_cluster"
    elif cluster_name_col in df_merged.columns:
        final_name_col = cluster_name_col
    else:
        final_name_col = "match_name"

    print("\nSample CSV names:")
    for x in df_merged[final_name_col].head(10):
        print(repr(x))

    print("\nSample PNG names from folder:")
    for x in sorted([p.name for p in boiling_plots_dir.glob("*.png")])[:10]:
        print(repr(x))

    exclude_cols = {cluster_col, "match_name"}
    X = build_feature_matrix(df_merged, exclude_cols=exclude_cols)

    keep_mask = ~X.isna().any(axis=1)
    dropped_for_na = int((~keep_mask).sum())

    df_merged = df_merged.loc[keep_mask].copy()
    X = X.loc[keep_mask].copy()

    if df_merged.empty:
        raise ValueError("All matched rows were dropped because of missing feature values.")

    print(f"\nFeature matrix shape after cleaning: {X.shape}")
    if dropped_for_na > 0:
        print(f"Dropped {dropped_for_na} rows with NaNs in feature columns.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=args.n_components,
        metric=args.metric,
        random_state=42
    )
    embedding = reducer.fit_transform(X_scaled)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        prediction_data=True
    )
    sub_labels = clusterer.fit_predict(embedding)

    df_merged["subcluster"] = sub_labels
    df_merged["sub_umap_1"] = embedding[:, 0]
    df_merged["sub_umap_2"] = embedding[:, 1]

    df_merged.to_csv(output_root / "rhythmic_subcluster_output.csv", index=False)

    counts = (
        df_merged["subcluster"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("subcluster")
        .reset_index(name="count")
    )
    counts.to_csv(output_root / "rhythmic_subcluster_counts.csv", index=False)

    feature_means = (
        pd.concat(
            [X.reset_index(drop=True), df_merged[["subcluster"]].reset_index(drop=True)],
            axis=1
        )
        .groupby("subcluster")
        .mean(numeric_only=True)
    )
    feature_means.to_csv(output_root / "rhythmic_subcluster_feature_means.csv")

    save_plots(embedding[:, :2], sub_labels, output_root)

    if not boiling_plots_dir.exists():
        raise FileNotFoundError(f"Could not find visuals folder: {boiling_plots_dir}")

    missing, copied = copy_matching_pngs(
        df_sub=df_merged,
        name_col=final_name_col,
        subcluster_col="subcluster",
        boiling_plots_dir=boiling_plots_dir,
        output_root=output_root
    )

    scores = compute_scores(embedding[:, :2], sub_labels)
    score_df = pd.DataFrame([scores])
    score_df.to_csv(output_root / "rhythmic_subcluster_scores.csv", index=False)

    with open(output_root / "run_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Main cluster CSV: {cluster_csv_path}\n")
        f.write(f"Feature CSV: {feature_csv_path}\n")
        f.write(f"Main cluster column: {cluster_col}\n")
        f.write(f"Target main cluster: {args.target_cluster}\n")
        f.write(f"Rows in target cluster before merge: {len(df_target)}\n")
        f.write(f"Rows matched with features: {len(df_merged)}\n")
        f.write(f"Feature count used: {X.shape[1]}\n")
        f.write(f"UMAP n_neighbors: {args.n_neighbors}\n")
        f.write(f"UMAP min_dist: {args.min_dist}\n")
        f.write(f"UMAP n_components: {args.n_components}\n")
        f.write(f"UMAP metric: {args.metric}\n")
        f.write(f"HDBSCAN min_cluster_size: {args.min_cluster_size}\n")
        f.write(f"HDBSCAN min_samples: {args.min_samples}\n")
        f.write(f"PNG copied: {copied}\n")
        f.write(f"PNG missing: {len(missing)}\n")
        f.write("\nSubcluster counts:\n")
        for _, row in counts.iterrows():
            f.write(f"  subcluster {row['subcluster']}: {row['count']}\n")

        f.write("\nSeparation scores:\n")
        for k, v in scores.items():
            f.write(f"  {k}: {v}\n")

        f.write("\nMissing PNG matches:\n")
        if missing:
            for m in missing:
                f.write(f"  {m}\n")
        else:
            f.write("  None\n")

    print("\nDone.")
    print(f"Outputs saved in: {output_root}")
    print(f"Copied PNGs: {copied}")
    print(f"Missing PNG matches: {len(missing)}")
    print("Separation scores:")
    for k, v in scores.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()