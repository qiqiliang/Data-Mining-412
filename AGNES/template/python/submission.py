# Submit this file to Gradescope
from typing import List
import math

class Solution:
    def euclidean_distance(self, point1, point2):
        """Calculate the Euclidean distance between two points."""
        return math.sqrt(sum((p - q) ** 2 for p, q in zip(point1, point2)))

    def find_clusters_to_merge(self, distance_matrix):
        """Identify the pair of clusters with the shortest distance between them."""
        min_distance = float('inf')
        clusters_to_merge = None
        for i, distances in enumerate(distance_matrix):
            for j, distance in enumerate(distances):
                if distance < min_distance and i != j:
                    min_distance = distance
                    clusters_to_merge = (i, j)
        return clusters_to_merge

    def update_distance_matrix(self, clusters, distance_matrix, linkage):
        """Update the distance matrix after merging clusters."""
        n = len(clusters)
        new_distances = [[float('inf') if i == j else 0 for j in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                distances = [self.euclidean_distance(p1, p2) for p1 in clusters[i] for p2 in clusters[j]]
                if linkage == 'single':
                    new_distances[i][j] = new_distances[j][i] = min(distances)
                elif linkage == 'complete':
                    new_distances[i][j] = new_distances[j][i] = max(distances)
                elif linkage == 'average':
                    new_distances[i][j] = new_distances[j][i] = sum(distances) / len(distances)
        return new_distances

    def agglomerative_clustering(self, X: List[List[float]], K: int, linkage: str) -> List[int]:
        """Generic agglomerative clustering function."""
        clusters = [[x] for x in X]
        distance_matrix = [[self.euclidean_distance(x, y) if x != y else float('inf') for y in X] for x in X]

        while len(clusters) > K:
            i, j = self.find_clusters_to_merge(distance_matrix)
            clusters[i] += clusters[j]
            del clusters[j]
            distance_matrix = self.update_distance_matrix(clusters, distance_matrix, linkage)

        labels = [None] * len(X)
        for idx, cluster in enumerate(clusters):
            for point in cluster:
                labels[X.index(point)] = idx
        return labels

    def hclus_single_link(self, X: List[List[float]], K: int) -> List[int]:
        """Perform hierarchical clustering using the single-link strategy."""
        return self.agglomerative_clustering(X, K, 'single')

    def hclus_average_link(self, X: List[List[float]], K: int) -> List[int]:
        """Perform hierarchical clustering using the average-link strategy."""
        return self.agglomerative_clustering(X, K, 'average')

    def hclus_complete_link(self, X: List[List[float]], K: int) -> List[int]:
        """Perform hierarchical clustering using the complete-link strategy."""
        return self.agglomerative_clustering(X, K, 'complete')

