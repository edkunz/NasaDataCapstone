# Generate code for DBScan algo
from sklearn.cluster import dbscan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def perform_dbscan(data: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform DBScan clustering on the given data.

    Parameters:
    - data: np.ndarray - The input data for clustering.
    - eps: float - The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_samples: int - The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
    - labels: np.ndarray - Cluster labels for each point in the dataset.
    - core_sample_indices: np.ndarray - Indices of core samples.
    """
    db = dbscan.DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(data)
    labels = db.labels_
    core_sample_indices = db.core_sample_indices_
    return labels, core_sample_indices
