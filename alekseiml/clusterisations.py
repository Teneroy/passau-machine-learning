import random

from base import Clusterisation
import numpy as np
from metrics import euclidean_distance


class KMeans(Clusterisation):
    def __init__(self, k: int = 1):
        self.centroids = None
        self.k = k

    def _do_k_means(self, X, k, distance_mesuring_method, movement_threshold_delta=0):
        new_centroids = self._get_initial_centroids(X=X, k=k)

        converged = False

        while not converged:
            previous_centroids = new_centroids
            clusters = self._compute_clusters(X, previous_centroids, distance_mesuring_method)

            new_centroids = np.array([np.mean(clusters[key], axis=0, dtype=X.dtype) for key in sorted(clusters.keys())])

            converged = self._check_convergence(previous_centroids, new_centroids, distance_mesuring_method,
                                                movement_threshold_delta)

        return new_centroids

    def _check_convergence(self, previous_centroids, new_centroids, distance_mesuring_method, movement_threshold_delta):
        distances_between_old_and_new_centroids = distance_mesuring_method(previous_centroids, new_centroids)
        converged = np.max(distances_between_old_and_new_centroids.diagonal()) <= movement_threshold_delta

        return converged

    def _compute_clusters(self, X, centroids, distance_mesuring_method):
        k = centroids.shape[0]
        clusters = {}
        distance_mat = distance_mesuring_method(X, centroids)
        closest_cluster_ids = np.argmin(distance_mat, axis=1)

        for i in range(k):
            clusters[i] = []

        for i, cluster_id in enumerate(closest_cluster_ids):
            clusters[cluster_id].append(X[i])

        return clusters

    def _get_initial_centroids(self, X, k):
        num_samples = X.shape[0]

        sample_pt_idx = random.sample(range(0, num_samples), k)

        centroids = [tuple(X[id]) for id in sample_pt_idx]
        unique_centroids = list(set(centroids))

        num_unique_centroids = len(unique_centroids)

        while num_unique_centroids < k:
            new_sample_pt_idx = random.sample(range(0, num_samples), k - num_unique_centroids)
            new_centroids = [tuple(X[id]) for id in new_sample_pt_idx]
            unique_centroids = list(set(unique_centroids + new_centroids))
            num_unique_centroids = len(unique_centroids)

        return np.array(unique_centroids)

    def learn(self, features: np.array, targets: np.array = None):
        self.centroids = self._do_k_means(features, k=self.k, distance_mesuring_method=euclidean_distance)
        return self

    def infer(self, features):
        distance_matrix = euclidean_distance(features, self.centroids)
        closest_cluster_ids = np.argmin(distance_matrix, axis=1)
        return closest_cluster_ids
