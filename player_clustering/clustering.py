import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import numpy as np
import umap.umap_ as umap
from sklearn.cluster import KMeans

from .embeddings import EmbeddingExtractor
from .fast_features import FastFeatureExtractor


class ClusteringManager:
    """
    Manager class for player clustering and team assignment.
    Handles UMAP dimensionality reduction and K-means clustering.
    """

    def __init__(self, n_components=3, n_clusters=2):
        """
        Initialize clustering models.

        Args:
            n_components: Number of components for UMAP reduction
            n_clusters: Number of clusters for K-means (typically 2 for teams)
        """
        self.reducer = umap.UMAP(n_components=n_components, random_state=42)
        self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.embedding_extractor = EmbeddingExtractor()
        self.fast_feature_extractor = FastFeatureExtractor()
        self.is_trained = False

    def project_embeddings(self, data, train=False):
        """
        Project embeddings to lower-dimensional space using UMAP.

        Args:
            data: High-dimensional embeddings
            train: Whether to fit the reducer or just transform

        Returns:
            Tuple of (reduced_embeddings, reducer)
        """
        if train:
            reduced_embeddings = self.reducer.fit_transform(data)
        else:
            reduced_embeddings = self.reducer.transform(data)

        return reduced_embeddings, self.reducer

    def cluster_embeddings(self, data, train=False):
        """
        Cluster embeddings using K-means.

        Args:
            data: Reduced embeddings for clustering
            train: Whether to fit the model or just predict

        Returns:
            Tuple of (cluster_labels, cluster_model)
        """
        if train:
            cluster_labels = self.cluster_model.fit_predict(data)
        else:
            cluster_labels = self.cluster_model.predict(data)

        return cluster_labels, self.cluster_model

    def process_batch(self, crop_batches, train=False):
        """
        Process a batch of crops through the full clustering pipeline.

        Args:
            crop_batches: Batched player crops
            train: Whether to train models or just predict

        Returns:
            Tuple of (cluster_labels, reducer, cluster_model)
        """
        # Extract embeddings
        embeddings = self.embedding_extractor.get_embeddings(crop_batches)

        # Reduce dimensionality
        reduced_embeddings, _ = self.project_embeddings(embeddings, train=train)

        # Cluster
        cluster_labels, _ = self.cluster_embeddings(reduced_embeddings, train=train)

        return cluster_labels, self.reducer, self.cluster_model


    def train_clustering_models(self, crops):
        """
        Train the UMAP and K-means models on player crops.
        OPTIMIZED: Uses fast color features instead of slow SigLIP embeddings.

        Args:
            crops: List of player crop images (PIL format)

        Returns:
            Tuple of (cluster_labels, reducer, cluster_model)
        """
        if crops is None or len(crops) == 0:
            raise ValueError("Crops list cannot be None or empty")

        print(f"Extracting fast color features from {len(crops)} training crops...")

        # Extract FAST color features from crops (compatible with inference)
        fast_features = self.fast_feature_extractor.get_features_from_crops(crops)

        print(f"Training UMAP + K-means on {len(fast_features)} samples...")

        # Project to lower-dimensional space using UMAP
        reduced_features = self.reducer.fit_transform(fast_features)

        # Cluster using K-means
        cluster_labels = self.cluster_model.fit_predict(reduced_features)

        # Mark as trained
        self.is_trained = True

        print(f"Training complete. Clusters: {len(set(cluster_labels))}, Feature dim: {fast_features.shape[1]}")

        return cluster_labels, self.reducer, self.cluster_model

    def get_cluster_labels(self, frame, player_detections, crops=None):
        """
        Get cluster labels for players in a single frame.
        FAST VERSION: Uses color histograms instead of slow SigLIP embeddings.

        Args:
            frame: Input video frame
            player_detections: Player detection results
            crops: Pre-extracted crops (optional, ignored in fast mode)

        Returns:
            Cluster labels for the players
        """
        if not self.is_trained:
            raise RuntimeError("Models must be trained before prediction. Call train_clustering_models() first.")

        if len(player_detections.xyxy) == 0:
            return np.array([])

        # Extract FAST color features (< 1ms per player vs 2600ms with SigLIP)
        fast_features = self.fast_feature_extractor.get_player_features(frame, player_detections)

        # Project features using trained UMAP
        reduced_features = self.reducer.transform(fast_features)

        # Predict clusters using trained K-means
        cluster_labels = self.cluster_model.predict(reduced_features)

        return cluster_labels