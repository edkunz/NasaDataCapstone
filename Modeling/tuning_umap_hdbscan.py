import os
import itertools
from datetime import datetime
from pathlib import Path
import re, ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import umap
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score


# Paths
FEATURES_CSV = "../data/features_updated.csv"
LABELS_XLSX  = "../data/Labeling.xlsx"

# output folder name
OUT_BASE = "../visuals/hparam_tunes"


# safe dict parser for new features
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


def load_features():
    df = pd.read_csv(FEATURES_CSV)
    file_names = df["file_name"].astype(str).values
    X = df.drop(columns=["file_name"])

    # expand dict-like columns
    dict_cols = [c for c in X.columns if X[c].astype(str).str.strip().str.startswith("{").any()]
    for c in dict_cols:
        parsed = X[c].apply(safe_parse_dict)
        expanded = pd.json_normalize(parsed).add_prefix(f"{c}_")
        X = X.drop(columns=[c]).join(expanded)

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return file_names, X_scaled


def load_labels():
    """
    Returns:
      gt_map: dict file_name -> label_int (1-8)
      gt_binary_map: dict file_name -> 0/1 (excluding transition)
    """
    gt_df = pd.read_excel(LABELS_XLSX)

    gt_df = gt_df.dropna(subset=["Label"]).copy()
    gt_df["Label"] = gt_df["Label"].astype(int)

    # match your features naming convention
    gt_df["file_name"] = "MATLAB " + gt_df["File"].astype(str)

    rhythmic_ids = [1, 2, 4, 6, 7]
    non_rhythmic_ids = [3, 5]

    def to_binary(lbl):
        if lbl in rhythmic_ids: return 1
        if lbl in non_rhythmic_ids: return 0
        return np.nan  # transition or anything else

    gt_df["ground_truth_binary"] = gt_df["Label"].apply(to_binary)

    # maps
    gt_map = dict(zip(gt_df["file_name"], gt_df["Label"]))
    gt_binary_map = dict(zip(
        gt_df.dropna(subset=["ground_truth_binary"])["file_name"],
        gt_df.dropna(subset=["ground_truth_binary"])["ground_truth_binary"].astype(int)
    ))

    return gt_map, gt_binary_map


def make_outdir():
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    outdir = Path(OUT_BASE) / f"tuning_analyze_{ts}"
    (outdir / "plots").mkdir(parents=True, exist_ok=True)
    return outdir


def plot_clusters(emb2d, labels, title, outpath):
    labels = np.asarray(labels)

    plt.figure(figsize=(8, 6))
    noise = labels == -1
    if noise.any():
        plt.scatter(emb2d[noise, 0], emb2d[noise, 1], s=6, alpha=0.35)

    for c in sorted([x for x in np.unique(labels) if x != -1]):
        m = labels == c
        plt.scatter(emb2d[m, 0], emb2d[m, 1], s=8, alpha=0.85)

    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def compute_metrics(emb2d, pred_labels, file_names, gt_map, gt_binary_map):
    pred_labels = np.asarray(pred_labels)

    # cluster counts
    clusters = [c for c in np.unique(pred_labels) if c != -1]
    n_clusters = len(clusters)
    noise_ratio = float((pred_labels == -1).mean())

    # silhouette (non-noise, requires >=2 clusters)
    sil = np.nan
    non_noise = pred_labels != -1
    if n_clusters >= 2 and non_noise.sum() >= 10:
        try:
            sil = float(silhouette_score(emb2d[non_noise], pred_labels[non_noise]))
        except Exception:
            sil = np.nan

    # ARI multi-class (exclude label 8 transitions by only using those in gt_map with label != 8)
    gt_multi = []
    pred_multi = []
    for fn, pl in zip(file_names, pred_labels):
        if fn in gt_map and gt_map[fn] != 8:
            gt_multi.append(gt_map[fn])
            pred_multi.append(pl)
    ari_multi = np.nan
    if len(set(pred_multi)) > 1 and len(gt_multi) > 10:
        ari_multi = float(adjusted_rand_score(gt_multi, pred_multi))

    # ARI binary (exclude transitions automatically because gt_binary_map doesn’t include them)
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
    Ranking: encourage meaningful subclusters + not too noisy.
    """
    sil = 0.0 if np.isnan(r["silhouette_non_noise"]) else r["silhouette_non_noise"]
    ari_multi = 0.0 if np.isnan(r["ARI_multiclass_1to7"]) else r["ARI_multiclass_1to7"]
    noise = r["noise_ratio"]
    k = r["n_clusters"]

    # prefer 3-12 clusters (substructure) but not fragmentation
    k_pen = 0.0
    if k < 3: k_pen = 0.7
    if k > 15: k_pen = 0.3 + 0.02*(k-15)

    noise_pen = max(0.0, noise - 0.35) * 1.2

    return 0.65*sil + 0.90*ari_multi - k_pen - noise_pen


def main():
    file_names, X_scaled = load_features()
    gt_map, gt_binary_map = load_labels()

    outdir = make_outdir()
    plots_dir = outdir / "plots"

    # Hyper-parameters
    UMAP_GRID = dict(
        n_neighbors=[7, 10, 15],
        min_dist=[0.0, 0.05, 0.1],
        metric=["euclidean"],
    )

    HDB_GRID = dict(
        min_cluster_size=[15, 20, 30, 40],
        min_samples=[None, 5, 10],
        cluster_selection_epsilon=[0.0, 0.05],
        cluster_selection_method=["eom"],
    )

    rows = []
    total = (len(UMAP_GRID["n_neighbors"]) * len(UMAP_GRID["min_dist"]) * len(UMAP_GRID["metric"])
             * len(HDB_GRID["min_cluster_size"]) * len(HDB_GRID["min_samples"])
             * len(HDB_GRID["cluster_selection_epsilon"]) * len(HDB_GRID["cluster_selection_method"]))
    print(f"Total combos: {total}")

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
            HDB_GRID["min_cluster_size"], HDB_GRID["min_samples"],
            HDB_GRID["cluster_selection_epsilon"], HDB_GRID["cluster_selection_method"]
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
                umap_n_neighbors=n_neighbors,
                umap_min_dist=min_dist,
                umap_metric=metric,
                hdb_min_cluster_size=mcs,
                hdb_min_samples=ms,
                hdb_epsilon=eps,
                hdb_selection=sel,
                **metrics
            )
            row["score"] = score_row(row)
            rows.append(row)


            # plot titles based on tuned params
            tag = (f"U_n{n_neighbors}_md{min_dist}_"
                f"__H_mcs{mcs}_ms{ms}_eps{eps}")
            simple_title = (
                f"UMAP: n_neighbors={n_neighbors}, min_dist={min_dist}\n"
                f"HDBSCAN: min_cluster_size={mcs}, min_samples={ms}, epsilon={eps}"
            )

            plot_clusters(emb, pred, simple_title, plots_dir / f"{tag}.png")

            if combo_i % 25 == 0:
                print(f"Progress {combo_i}/{total}")

    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    df.to_csv(outdir / "metrics.csv", index=False)

    top = df.head(20)
    (outdir / "top_ranked.txt").write_text(top.to_string(index=False))
    #Save images
    export_clusters_for_selected_tunings(outdir, file_names, X_scaled)

    print(f"\nSaved everything to: {outdir}")
    print("\nTop 10 combos:")
    print(df.head(10)[[
        "score","n_clusters","noise_ratio","silhouette_non_noise","ARI_multiclass_1to7",
        "umap_n_neighbors","umap_min_dist","hdb_min_cluster_size","hdb_min_samples","hdb_epsilon"
    ]])


# Cluster PNG export helpers by tuning setting
import shutil

def _sanitize(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(s))

def find_png_for_file(file_name: str, source_dirs):
    """
    Try to find a PNG for a given file_name by searching under source_dirs.
    We match either exact stem or substring match (common when file_name has spaces).
    """
    target = _sanitize(file_name)
    candidates = []

    for d in source_dirs:
        d = Path(d)
        if not d.exists():
            continue

        # Fast path: exact match
        exact = list(d.rglob(f"{target}.png"))
        if exact:
            return exact[0]

        # Broader: any PNG containing the sanitized filename
        for p in d.rglob("*.png"):
            if target in _sanitize(p.stem):
                candidates.append(p)

    # If multiple matches, return the shortest path (usually most direct)
    if candidates:
        candidates = sorted(candidates, key=lambda x: len(str(x)))
        return candidates[0]

    return None


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
    """
    Recompute UMAP embedding + HDBSCAN clustering for a single tuning and export PNGs into:
      outdir/cluster_exports/<tuning_name>/cluster_<id>/
    """
    export_root = Path(outdir) / "cluster_exports" / tuning_name
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

    # Save assignment table
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

    # Copy PNGs into cluster folders
    for fn, c in zip(file_names, labels):
        png_path = find_png_for_file(fn, source_png_dirs)
        if png_path is None:
            missing.append(fn)
            continue

        dest = export_root / f"cluster_{c}" / png_path.name
        try:
            shutil.copy2(png_path, dest)
        except Exception:
            # If duplicate names collide, include sanitized filename
            dest = export_root / f"cluster_{c}" / f"{_sanitize(fn)}__{png_path.name}"
            shutil.copy2(png_path, dest)

    # Report missing
    if missing:
        (export_root / "missing_pngs.txt").write_text("\n".join(map(str, missing)))

    print(f"[export] {tuning_name}: wrote {len(assign_df)} assignments to {export_root}")
    print(f"[export] {tuning_name}: missing PNGs for {len(missing)} files")


def export_clusters_for_selected_tunings(outdir: Path, file_names, X_scaled):
    """
    Edit these to the 3 tunings you care about.
    These are the 3 common candidates you were comparing:
      1) nn=10, md=0.1, mcs=30, ms=5, eps=0.0
      2) nn=15, md=0.05, mcs=15, ms=5, eps=0.0
      3) nn=7,  md=0.05, mcs=15, ms=5, eps=0.0
    """

    SOURCE_PNG_DIRS = [
        "../data/CSV"
    ]

    SELECTED = [
        dict(
            tuning_name="UMAP_nn10_md0.10__HDB_mcs30_ms5_eps0.00",
            umap_params=dict(n_neighbors=10, min_dist=0.10, metric="euclidean", random_state=42),
            hdb_params=dict(min_cluster_size=30, min_samples=5, cluster_selection_epsilon=0.00, cluster_selection_method="eom"),
        ),
        dict(
            tuning_name="UMAP_nn15_md0.05__HDB_mcs15_ms5_eps0.00",
            umap_params=dict(n_neighbors=15, min_dist=0.05, metric="euclidean", random_state=42),
            hdb_params=dict(min_cluster_size=15, min_samples=5, cluster_selection_epsilon=0.00, cluster_selection_method="eom"),
        ),
        dict(
            tuning_name="UMAP_nn7_md0.05__HDB_mcs15_ms5_eps0.00",
            umap_params=dict(n_neighbors=7, min_dist=0.05, metric="euclidean", random_state=42),
            hdb_params=dict(min_cluster_size=15, min_samples=5, cluster_selection_epsilon=0.00, cluster_selection_method="eom"),
        ),
    ]

    for item in SELECTED:
        export_cluster_pngs_for_tuning(
            outdir=Path(outdir),
            file_names=file_names,
            X_scaled=X_scaled,
            tuning_name=item["tuning_name"],
            umap_params=item["umap_params"],
            hdb_params=item["hdb_params"],
            source_png_dirs=SOURCE_PNG_DIRS,
        )


if __name__ == "__main__":
    main()