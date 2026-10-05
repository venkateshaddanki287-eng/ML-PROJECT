"""
CO4: Unsupervised Learning & Dimensionality Reduction.
Implements K-Means, Agglomerative Hierarchical Clustering, DBSCAN,
PCA, t-SNE, and UMAP (with fallback).
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


class UnsupervisedTrainer:
    """
    Performs clustering and dimensionality reduction on traffic feature datasets.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def fit_kmeans(self, X: pd.DataFrame, n_clusters: int = 3) -> Tuple[KMeans, np.ndarray, pd.DataFrame]:
        """
        Fits K-Means clustering and analyzes cluster profiles.
        Labels are not hardcoded before clustering.
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        # Profile centroids in original feature scale
        centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X.columns)
        centroids['Cluster_Size'] = pd.Series(labels).value_counts().sort_index().values
        
        # Determine natural cluster interpretations based on total queue centroids
        queue_cols = [c for c in ['north_queue', 'south_queue', 'east_queue', 'west_queue'] if c in centroids.columns]
        centroids['Mean_Total_Queue'] = centroids[queue_cols].sum(axis=1)
        
        # Rank clusters by total queue
        sorted_indices = centroids['Mean_Total_Queue'].argsort()
        label_map = {sorted_indices.iloc[0]: 'Low Traffic', sorted_indices.iloc[1]: 'Medium Traffic', sorted_indices.iloc[2]: 'Heavy Traffic'}
        centroids['Interpretation'] = [label_map.get(i, f'Cluster {i}') for i in range(n_clusters)]

        return kmeans, labels, centroids

    def fit_hierarchical(self, X: pd.DataFrame, n_clusters: int = 3) -> Tuple[AgglomerativeClustering, np.ndarray]:
        """Fits Agglomerative Hierarchical Clustering."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        agg = AgglomerativeClustering(n_clusters=n_clusters)
        labels = agg.fit_predict(X_scaled)
        return agg, labels

    def fit_dbscan(self, X: pd.DataFrame, eps: float = 1.2, min_samples: int = 10) -> Tuple[DBSCAN, np.ndarray, Dict[str, int]]:
        """
        Fits DBSCAN to identify core traffic patterns vs outliers (label -1).
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(list(labels).count(-1))

        stats = {
            'n_clusters': n_clusters,
            'n_outliers': n_noise,
            'outlier_percentage': round(100.0 * n_noise / len(X), 2)
        }
        return dbscan, labels, stats

    def run_pca(self, X: pd.DataFrame, n_components: int = 2) -> Tuple[PCA, np.ndarray, Dict[str, Any]]:
        """Performs PCA dimensionality reduction."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=n_components, random_state=self.random_state)
        coords = pca.fit_transform(X_scaled)

        stats = {
            'explained_variance_ratio': [round(float(v), 4) for v in pca.explained_variance_ratio_],
            'cumulative_explained_variance': round(float(np.sum(pca.explained_variance_ratio_)), 4)
        }
        return pca, coords, stats

    def run_tsne(self, X: pd.DataFrame, n_components: int = 2, perplexity: float = 30.0) -> np.ndarray:
        """Performs t-SNE visualization mapping."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Subsample to max 1000 points for fast execution if dataset is huge
        if len(X_scaled) > 1000:
            idx = np.random.choice(len(X_scaled), size=1000, replace=False)
            X_sample = X_scaled[idx]
        else:
            X_sample = X_scaled

        tsne = TSNE(n_components=n_components, perplexity=min(perplexity, len(X_sample)-1), random_state=self.random_state)
        coords = tsne.fit_transform(X_sample)
        return coords

    def run_umap(self, X: pd.DataFrame, n_components: int = 2) -> Tuple[np.ndarray, bool]:
        """
        Performs UMAP dimensionality reduction with fallback to PCA if UMAP is unavailable.
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if HAS_UMAP:
            reducer = umap.UMAP(n_components=n_components, random_state=self.random_state)
            coords = reducer.fit_transform(X_scaled)
            return coords, True
        else:
            pca = PCA(n_components=n_components, random_state=self.random_state)
            coords = pca.fit_transform(X_scaled)
            return coords, False
