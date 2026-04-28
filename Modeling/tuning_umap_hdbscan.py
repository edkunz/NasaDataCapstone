import itertools
from datetime import datetime
from pathlib import Path
import re
import ast
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import umap
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score


# -----------------------------
# Project-relative paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# keep your changed features file here
FEATURES_CSV = PROJECT_ROOT / "data" / "non_noise_features.csv"
LABELS_XLSX = PROJECT_ROOT / "data" / "Labeling.xlsx"

# these two are only used to choose which runs belong to active boiling
CLUSTER_1_CSV = PROJECT_ROOT / "visuals" / "rep_samples" / "UMAP-HDBSCAN" / "1" / "cluster_1.csv"
CLUSTER_2_CSV = PROJECT_ROOT / "visuals" / "rep_samples" / "UMAP-HDBSCAN" / "2" / "cluster_2.csv"

OUT_BASE = PROJECT_ROOT / "visuals" / "hparam_tunes"


# -----------------------------
# Helpers
# -----------------------------
def safe_parse_dict(s):
    if pd.isna(s):
        return {}
    s = str(s).strip()
    if not s.startswith("{"):
        return {}

    s = re.sub(r"np\.float64\(([^)]+)\)", r"\1", s)
    s = s.replace("nan", "None")

    try:
        out = ast.literal_eval(s)
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def make_outdir():
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    outdir = OUT_BASE / f"tuning_active_boiling_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def plot_clusters(emb2d, labels, title, outpath):
    labels = np.asarray(labels)

    plt.figure(figsize=(8, 6))
    noise = labels == -1
    if noise.any():
        plt.scatter(emb2d[noise, 0], emb2d[noise, 1], s=8, alpha=0.35, label="Noise")

    for c in sorted([x for x in np.unique(labels) if x != -1]):
        m = labels == c
        plt.scatter(emb2d[m, 0], emb2d[m, 1], s=10, alpha=0.85, label=f"Cluster {c}")

    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def _sanitize(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(s))


def find_png_for_file(file_name: str, source_dirs):
    target = _sanitize(file_name)
    candidates = []

    for d in source_dirs:
        d = Path(d)
        if not d.exists():
            continue

        exact = list(d.rglob(f"{target}.png"))
        if exact:
            return exact[0]

        for p in d.rglob("*.png"):
            if target in _sanitize(p.stem):
                candidates.append(p)

    if candidates:
        candidates = sorted(candidates, key=lambda x: len(str(x)))
        return candidates[0]

    return None


# -----------------------------
# Data loading
# -----------------------------
def load_cluster_file_names(cluster_csv_path):
    df = pd.read_csv(cluster_csv_path)

    if "file_name" in df.columns:
        names = df["file_name"].astype(str).tolist()
    elif "name" in df.columns:
        names = df["name"].astype(str).tolist()
    else:
        raise ValueError(f"Expected 'file_name' or 'name' column in {cluster_csv_path}")

    seen = set()
    ordered_unique = []
    for n in names:
        if n not in seen:
            seen.add(n)
            ordered_unique.append(n)

    return ordered_unique


def load_combined_active_boiling_file_names():
    cluster_1_names = load_cluster_file_names(CLUSTER_1_CSV)
    cluster_2_names = load_cluster_file_names(CLUSTER_2_CSV)

    combined = cluster_1_names + cluster_2_names

    seen = set()
    ordered_unique = []
    for n in combined:
        if n not in seen:
            seen.add(n)
            ordered_unique.append(n)

    return ordered_unique


def load_features_for_selected_runs(selected_file_names):
    """
    Uses ONLY non_noise_features.csv as the feature source.
    Selected file_names determine which rows to keep.
    """
    df = pd.read_csv(FEATURES_CSV)

    if "file_name" not in df.columns:
        raise ValueError("Expected 'file_name' column in non_noise_features.csv")

    df["file_name"] = df["file_name"].astype(str)
    df = df[df["file_name"].isin(selected_file_names)].copy()

    order_map = {fn: i for i, fn in enumerate(selected_file_names)}
    df["__order"] = df["file_name"].map(order_map)
    df = df.sort_values("__order").drop(columns="__order")

    file_names = df["file_name"].astype(str).values
    X = df.drop(columns=["file_name"]).copy()

    # expand dict-like columns if any exist
    dict_cols = [c for c in X.columns if X[c].astype(str).str.strip().str.startswith("{").any()]
    for c in dict_cols:
        parsed = X[c].apply(safe_parse_dict)
        expanded = pd.json_normalize(parsed).add_prefix(f"{c}_")
        X = X.drop(columns=[c]).join(expanded)

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return file_names, X, X_scaled


def load_labels():
    gt_df = pd.read_excel(LABELS_XLSX)

    gt_df = gt_df.dropna(subset=["Label"]).copy()
    gt_df["Label"] = gt_df["Label"].astype(int)

    gt_df["file_name"] = "MATLAB " + gt_df["File"].astype(str)

    rhythmic_ids = [1, 2, 4, 6, 7]
    non_rhythmic_ids = [3, 5]

    def to_binary(lbl):
        if lbl in rhythmic_ids:
            return 1
        if lbl in non_rhythmic_ids:
            return 0
        return np.nan

    gt_df["ground_truth_binary"] = gt_df["Label"].apply(to_binary)

    gt_map = dict(zip(gt_df["file_name"], gt_df["Label"]))
    gt_binary_map = dict(zip(
        gt_df.dropna(subset=["ground_truth_binary"])["file_name"],
        gt_df.dropna(subset=["ground_truth_binary"])["ground_truth_binary"].astype(int)
    ))

    return gt_map, gt_binary_map


# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(emb2d, pred_labels, file_names, gt_map, gt_binary_map):
    pred_labels = np.asarray(pred_labels)

    clusters = [c for c in np.unique(pred_labels) if c != -1]
    n_clusters = len(clusters)
    noise_ratio = float((pred_labels == -1).mean())

    sil = np.nan
    non_noise = pred_labels != -1
    if n_clusters >= 2 and non_noise.sum() >= 10:
        try:
            sil = float(silhouette_score(emb2d[non_noise], pred_labels[non_noise]))
        except Exception:
            sil = np.nan

    gt_multi = []
    pred_multi = []
    for fn, pl in zip(file_names, pred_labels):
        if fn in gt_map and gt_map[fn] != 8:
            gt_multi.append(gt_map[fn])
            pred_multi.append(pl)

    ari_multi = np.nan
    if len(set(pred_multi)) > 1 and len(gt_multi) > 10:
        ari_multi = float(adjusted_rand_score(gt_multi, pred_multi))

    gt_bin = []
    pred_bin = []
    for fn, pl in zip(file_names, pred_labels):
        if fn in gt_binary_map:
            gt_bin.append(gt_binary_map[fn])
            pred_bin.append(pl)

    ari_bin = np.nan
    if len(set(pred_bin)) > 1 and len(gt_bin) > 10:
        ari_bin = float(adjusted_rand_score(gt_bin, pred_bin))

    return {
        "n_clusters": n_clusters,
        "noise_ratio": noise_ratio,
        "silhouette_non_noise": sil,
        "ARI_multiclass_1to7": ari_multi,
        "ARI_binary": ari_bin,
        "n_labeled_multiclass": len(gt_multi),
        "n_labeled_binary": len(gt_bin),
    }


def score_row(r):
    """
    Prefer moderate subcluster counts and decent separation.
    """
    sil = 0.0 if np.isnan(r["silhouette_non_noise"]) else r["silhouette_non_noise"]
    ari_multi = 0.0 if np.isnan(r["ARI_multiclass_1to7"]) else r["ARI_multiclass_1to7"]
    noise = r["noise_ratio"]
    k = r["n_clusters"]

    # prefer roughly 4-12 clusters
    k_pen = 0.0
    if k < 4:
        k_pen = 0.8
    elif k > 14:
        k_pen = 0.25 + 0.04 * (k - 14)

    noise_pen = max(0.0, noise - 0.30) * 1.4

    return 0.75 * sil + 0.50 * ari_multi - k_pen - noise_pen


# -----------------------------
# Optional cluster export helpers
# -----------------------------
def export_cluster_pngs_for_tuning(
    *,
    outdir: Path,
    file_names: np.ndarray,
    X_scaled: np.ndarray,
    tuning_name: str,
    umap_params: dict,
    hdb_params: dict,
    source_png_dirs: list,
):
    export_root = outdir / "cluster_exports" / tuning_name
    export_root.mkdir(parents=True, exist_ok=True)

    reducer = umap.UMAP(
        n_neighbors=umap_params["n_neighbors"],
        min_dist=umap_params["min_dist"],
        n_components=2,
        metric=umap_params.get("metric", "euclidean"),
        random_state=umap_params.get("random_state", 42),
    )
    emb = reducer.fit_transform(X_scaled)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=hdb_params["min_cluster_size"],
        min_samples=hdb_params.get("min_samples", None),
        metric=hdb_params.get("metric", "euclidean"),
        cluster_selection_epsilon=hdb_params.get("cluster_selection_epsilon", 0.0),
        cluster_selection_method=hdb_params.get("cluster_selection_method", "eom"),
    )
    labels = clusterer.fit_predict(emb)

    assign_df = pd.DataFrame({
        "file_name": file_names,
        "cluster": labels,
        "umap_n_neighbors": umap_params["n_neighbors"],
        "umap_min_dist": umap_params["min_dist"],
        "hdb_min_cluster_size": hdb_params["min_cluster_size"],
        "hdb_min_samples": hdb_params.get("min_samples", None),
        "hdb_epsilon": hdb_params.get("cluster_selection_epsilon", 0.0),
        "hdb_selection": hdb_params.get("cluster_selection_method", "eom"),
    })
    assign_df.to_csv(export_root / "cluster_assignments.csv", index=False)

    missing = []
    unique_clusters = sorted(set(labels.tolist()))
    for c in unique_clusters:
        (export_root / f"cluster_{c}").mkdir(parents=True, exist_ok=True)

    for fn, c in zip(file_names, labels):
        png_path = find_png_for_file(fn, source_png_dirs)
        if png_path is None:
            missing.append(fn)
            continue

        dest = export_root / f"cluster_{c}" / png_path.name
        try:
            shutil.copy2(png_path, dest)
        except Exception:
            dest = export_root / f"cluster_{c}" / f"{_sanitize(fn)}__{png_path.name}"
            shutil.copy2(png_path, dest)

    if missing:
        (export_root / "missing_pngs.txt").write_text("\n".join(map(str, missing)))

    print(f"[export] {tuning_name}: wrote {len(assign_df)} assignments to {export_root}")
    print(f"[export] {tuning_name}: missing PNGs for {len(missing)} files")


# -----------------------------
# Tuning run for combined active boiling
# -----------------------------
def run_tuning_for_group(group_name, selected_file_names, parent_outdir, gt_map, gt_binary_map):
    outdir = parent_outdir / f"{group_name}_tuning"
    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n===== Running tuning for {group_name} =====")
    print(f"Selected runs: {len(selected_file_names)}")

    file_names, X_df, X_scaled = load_features_for_selected_runs(selected_file_names)

    UMAP_GRID = dict(
        n_neighbors=[10, 15, 20, 25, 30],
        min_dist=[0.0, 0.03, 0.05, 0.08, 0.12],
        metric=["euclidean"],
    )

    HDB_GRID = dict(
        min_cluster_size=[8, 10, 12, 15, 18, 20],
        min_samples=[5, 8, 10, 12],
        cluster_selection_epsilon=[0.0, 0.03, 0.05],
        cluster_selection_method=["eom"],
    )

    rows = []
    total = (
        len(UMAP_GRID["n_neighbors"])
        * len(UMAP_GRID["min_dist"])
        * len(UMAP_GRID["metric"])
        * len(HDB_GRID["min_cluster_size"])
        * len(HDB_GRID["min_samples"])
        * len(HDB_GRID["cluster_selection_epsilon"])
        * len(HDB_GRID["cluster_selection_method"])
    )
    print(f"Total combos for {group_name}: {total}")

    combo_i = 0
    for n_neighbors, min_dist, metric in itertools.product(
        UMAP_GRID["n_neighbors"], UMAP_GRID["min_dist"], UMAP_GRID["metric"]
    ):
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            metric=metric,
            random_state=42,
        )
        emb = reducer.fit_transform(X_scaled)

        for mcs, ms, eps, sel in itertools.product(
            HDB_GRID["min_cluster_size"],
            HDB_GRID["min_samples"],
            HDB_GRID["cluster_selection_epsilon"],
            HDB_GRID["cluster_selection_method"],
        ):
            combo_i += 1

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=mcs,
                min_samples=ms,
                metric="euclidean",
                cluster_selection_epsilon=eps,
                cluster_selection_method=sel,
            )
            pred = clusterer.fit_predict(emb)

            metrics = compute_metrics(emb, pred, file_names, gt_map, gt_binary_map)

            row = dict(
                group_name=group_name,
                umap_n_neighbors=n_neighbors,
                umap_min_dist=min_dist,
                umap_metric=metric,
                hdb_min_cluster_size=mcs,
                hdb_min_samples=ms,
                hdb_epsilon=eps,
                hdb_selection=sel,
                **metrics,
            )
            row["score"] = score_row(row)
            rows.append(row)

            tag = f"U_n{n_neighbors}_md{min_dist}__H_mcs{mcs}_ms{ms}_eps{eps}"
            simple_title = (
                f"{group_name}\n"
                f"UMAP: n_neighbors={n_neighbors}, min_dist={min_dist}\n"
                f"HDBSCAN: min_cluster_size={mcs}, min_samples={ms}, epsilon={eps}"
            )

            n_clusters = len(set(pred)) - (1 if -1 in pred else 0)
            cluster_dir = plots_dir / f"clusters_{n_clusters}"
            cluster_dir.mkdir(parents=True, exist_ok=True)

            plot_clusters(
                emb,
                pred,
                simple_title,
                cluster_dir / f"{tag}.png"
            )

            if combo_i % 25 == 0:
                print(f"{group_name}: progress {combo_i}/{total}")

    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    df.to_csv(outdir / "metrics.csv", index=False)

    top = df.head(25)
    (outdir / "top_ranked.txt").write_text(top.to_string(index=False))

    print(f"\nSaved {group_name} outputs to: {outdir}")
    print(f"\nTop 10 combos for {group_name}:")
    print(df.head(10)[[
        "score",
        "n_clusters",
        "noise_ratio",
        "silhouette_non_noise",
        "ARI_multiclass_1to7",
        "umap_n_neighbors",
        "umap_min_dist",
        "hdb_min_cluster_size",
        "hdb_min_samples",
        "hdb_epsilon",
    ]])

    return df


# -----------------------------
# Main
# -----------------------------
def main():
    outdir = make_outdir()
    gt_map, gt_binary_map = load_labels()

    combined_names = load_combined_active_boiling_file_names()

    print(f"combined active boiling selected runs: {len(combined_names)}")
    print(f"Root output folder: {outdir}")

    df_combined = run_tuning_for_group(
        group_name="combined_cluster_1_2",
        selected_file_names=combined_names,
        parent_outdir=outdir,
        gt_map=gt_map,
        gt_binary_map=gt_binary_map,
    )

    print("\nDone.")
    print(f"All outputs saved under: {outdir}")


if __name__ == "__main__":
    main()