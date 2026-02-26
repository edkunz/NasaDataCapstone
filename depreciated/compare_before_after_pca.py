import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN


def load_features(csv_path: str):
    """Load a features CSV and return (file_names, numeric_matrix, numeric_col_names)."""
    df = pd.read_csv(csv_path)

    if "file_name" in df.columns:
        file_names = df["file_name"].astype(str).copy()
        numeric_cols = [c for c in df.columns if c != "file_name"]
    else:
        # If file_name doesn't exist, make a dummy series
        file_names = pd.Series([f"row_{i}" for i in range(len(df))])
        numeric_cols = list(df.columns)

    # Keep only numeric columns and replace NaNs/infs
    X = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return file_names, X.values, numeric_cols


def scale_and_pca(X: np.ndarray, n_components: int = 6):
    """Standardize features and apply PCA. Returns (X_scaled, X_pca, pca_model)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=0)
    X_pca = pca.fit_transform(X_scaled)

    return X_scaled, X_pca, pca


def plot_pca_2d(X_pca: np.ndarray, title: str, out_path: str, labels=None):
    """Scatter plot PC1 vs PC2. If labels are provided, color by label."""
    plt.figure(figsize=(8, 6))

    if labels is None:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], s=18)
    else:
        labels = np.asarray(labels)
        # Plot each label separately so matplotlib auto-colors them
        for lab in np.unique(labels):
            mask = labels == lab
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], s=18, label=str(lab))
        plt.legend(title="cluster", fontsize=8)

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    before_csv = os.path.join("data", "features_before.csv")
    after_csv = os.path.join("data", "features.csv")

    if not os.path.exists(before_csv):
        raise FileNotFoundError(f"Missing: {before_csv}")
    if not os.path.exists(after_csv):
        raise FileNotFoundError(f"Missing: {after_csv}")

    # ---- BEFORE (cross accelerometer features added) ----
    _, X_before, _ = load_features(before_csv)
    _, Xp_before, pca_before = scale_and_pca(X_before, n_components=6)
    print(f"BEFORE explained variance (PC1..PC6): {np.round(pca_before.explained_variance_ratio_, 4)}")

    plot_pca_2d(
        Xp_before,
        title="BEFORE: PCA (scaled) — PC1 vs PC2",
        out_path=os.path.join("data", "pca_before.png")
    )

    # DBSCAN on BEFORE PCA space (same idea as “cluster then plot”)
    db_before = DBSCAN(eps=0.8, min_samples=5).fit(Xp_before[:, :2])
    plot_pca_2d(
        Xp_before,
        title="BEFORE: PCA + DBSCAN labels (PC1 vs PC2)",
        out_path=os.path.join("data", "pca_before_dbscan.png"),
        labels=db_before.labels_
    )

    # ---- AFTER ----
    _, X_after, _ = load_features(after_csv)
    _, Xp_after, pca_after = scale_and_pca(X_after, n_components=6)
    print(f"AFTER explained variance (PC1..PC6): {np.round(pca_after.explained_variance_ratio_, 4)}")

    plot_pca_2d(
        Xp_after,
        title="AFTER: PCA (scaled) — PC1 vs PC2",
        out_path=os.path.join("data", "pca_after.png")
    )

    db_after = DBSCAN(eps=0.8, min_samples=5).fit(Xp_after[:, :2])
    plot_pca_2d(
        Xp_after,
        title="AFTER: PCA + DBSCAN labels (PC1 vs PC2)",
        out_path=os.path.join("data", "pca_after_dbscan.png"),
        labels=db_after.labels_
    )

    print("Saved plots:")
    print("  data/pca_before.png")
    print("  data/pca_before_dbscan.png")
    print("  data/pca_after.png")
    print("  data/pca_after_dbscan.png")


if __name__ == "__main__":
    main()
