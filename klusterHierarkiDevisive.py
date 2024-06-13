import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

class DivisiveClustering:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit_predict(self, X):
        n_samples = X.shape[0]
        distances = pairwise_distances(X)
        np.fill_diagonal(distances, np.inf)  # Exclude diagonal elements

        labels = np.zeros(n_samples)
        current_label = 0

        while current_label < self.n_clusters:
            max_distance_index = np.argmax(np.sum(distances, axis=1))
            cluster_indices = np.where(labels == current_label)[0]

            # Find the furthest point in the current cluster
            furthest_point_index = cluster_indices[np.argmax(distances[cluster_indices, max_distance_index])]

            # Assign the furthest point to a new cluster
            labels[furthest_point_index] = current_label + 1

            # Update distances matrix by excluding points already assigned to the new cluster
            distances[furthest_point_index, cluster_indices] = np.inf
            distances[cluster_indices, furthest_point_index] = np.inf

            current_label += 1

        return labels

# Example usage
np.random.seed(0)
X = np.random.randn(10, 2)

# Perform divisive clustering
divisive = DivisiveClustering(n_clusters=2)
labels = divisive.fit_predict(X)

print("Cluster labels:", labels)

# Calculate centroids of the clusters
unique_labels = np.unique(labels)
centroids = np.array([X[labels == label].mean(axis=0) for label in unique_labels])

# Print the coordinates of the centroids
for i, centroid in enumerate(centroids):
    print(f"Centroid {i}: {centroid}")

# Visualize the clustering
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='*', s=200, edgecolor='black')
plt.title('Divisive Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Generate the linkage matrix for the dendrogram
linkage_matrix = linkage(X, method='ward')

# Visualize the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()
