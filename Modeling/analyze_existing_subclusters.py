from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import zscore


def find_column(df, candidates, required=False):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise ValueError(f"Could not find any of these columns: {candidates}")
    return None


def guess_name_column(df):
    candidates = ["name", "run_name", "run", "filename", "file_name", "png_name", "plot_name", "match_name"]
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
    s = s.replace('"', "").replace("'", "")
    if s.lower().endswith(".csv"):
        s = s[:-4]
    if s.lower().endswith(".png"):
        s = s[:-4]
    s = s.replace(" .", ".")
    s = " ".join(s.split())
    return s.strip()


def main():
    project_root = Path(__file__).resolve().parent.parent

    # CHANGE THESE ONLY IF NEEDED
    subcluster_csv = project_root / "data" / "rhythmic_subclusters" / "rhythmic_subcluster_output.csv"
    feature_csv = project_root / "data" / "features.csv"
    output_dir = project_root / "data" / "rhythmic_subclusters"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading subcluster file: {subcluster_csv}")
    df_sub = pd.read_csv(subcluster_csv)

    print(f"Reading feature file: {feature_csv}")
    df_feat = pd.read_csv(feature_csv)

    sub_name_col = guess_name_column(df_sub)
    feat_name_col = guess_name_column(df_feat)

    if "subcluster" not in df_sub.columns:
        raise ValueError("The subcluster output file must contain a 'subcluster' column.")

    df_sub["match_name"] = df_sub[sub_name_col].apply(clean_run_name)
    df_feat["match_name"] = df_feat[feat_name_col].apply(clean_run_name)

    df = df_sub.merge(df_feat, on="match_name", how="inner", suffixes=("_sub", "_feat"))

    if df.empty:
        raise ValueError("Merge found 0 matching rows between subcluster file and features file.")

    # remove non-feature numeric columns
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    drop_cols = []
    for c in numeric_df.columns:
        lc = c.lower()
        if c == "subcluster":
            continue
        if "cluster" in lc or "subcluster" in lc or "umap" in lc or "prob" in lc or "score" in lc or "membership" in lc:
            drop_cols.append(c)

    feature_cols = [c for c in numeric_df.columns if c not in drop_cols and c != "subcluster"]

    X = df[feature_cols].copy()
    y = df["subcluster"].copy()

    keep_mask = ~X.isna().any(axis=1)
    X = X.loc[keep_mask].copy()
    y = y.loc[keep_mask].copy()
    df = df.loc[keep_mask].copy()

    if X.empty:
        raise ValueError("No usable feature rows remain after dropping NaNs.")

    X_z = pd.DataFrame(
        zscore(X, nan_policy="omit"),
        columns=X.columns,
        index=X.index
    )

    df_z = pd.concat([X_z.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    df_z.rename(columns={0: "subcluster"}, inplace=True)

    # counts
    counts = (
        y.value_counts()
        .sort_index()
        .rename_axis("subcluster")
        .reset_index(name="count")
    )
    counts.to_csv(output_dir / "subcluster_counts_checked.csv", index=False)

    # means
    feature_means = (
        pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        .groupby("subcluster")
        .mean(numeric_only=True)
    )
    feature_means.to_csv(output_dir / "subcluster_feature_means_checked.csv")
    
    # check the standard deviation of features within clusters
    feature_stds = (
        pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        .groupby("subcluster")
        .std(numeric_only=True)
    )
    feature_stds.to_csv(output_dir / "subcluster_feature_stds_checked.csv")

    # top differentiating features per cluster
    feature_diff_rows = []

    all_feature_names = X.columns.tolist()

    for c in sorted(df_z["subcluster"].unique()):
        cluster_only = df_z[df_z["subcluster"] == c][all_feature_names]
        rest_only = df_z[df_z["subcluster"] != c][all_feature_names]

        cluster_mean = cluster_only.mean()
        rest_mean = rest_only.mean()

        diff = (cluster_mean - rest_mean).abs().sort_values(ascending=False)

        for rank, (feature_name, score) in enumerate(diff.head(10).items(), start=1):
            feature_diff_rows.append({
                "subcluster": int(c),
                "rank": rank,
                "feature": feature_name,
                "abs_z_mean_diff_vs_rest": float(score),
                "cluster_z_mean": float(cluster_mean[feature_name]),
                "rest_z_mean": float(rest_mean[feature_name])
            })

    feature_diff_df = pd.DataFrame(feature_diff_rows)
    feature_diff_df.to_csv(output_dir / "subcluster_feature_importance_checked.csv", index=False)

    # outlier cluster by size
    cluster_sizes = counts.set_index("subcluster")["count"]
    threshold = cluster_sizes.mean() * 0.3
    outliers = (
        cluster_sizes[cluster_sizes < threshold]
        .rename_axis("subcluster")
        .reset_index(name="count")
    )
    outliers["threshold"] = float(threshold)
    outliers.to_csv(output_dir / "outlier_clusters_checked.csv", index=False)

    print("\nDone.")
    print(f"Rows analyzed: {len(df)}")
    print(f"Feature count: {len(feature_cols)}")
    print(f"Output folder: {output_dir}")


if __name__ == "__main__":
    main()