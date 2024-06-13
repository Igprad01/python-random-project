import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Generate sample data from sklearn.datasets
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Perform hierarchical clustering
clustering = AgglomerativeClustering(n_clusters=4).fit(X)

# Calculate centroids of the clusters
centroids = np.array([X[clustering.labels_ == i].mean(axis=0) for i in range(4)])

# Print the coordinates of the centroids
centroid_coordinates = {f"Centroid {i}": centroid for i, centroid in enumerate(centroids)}
print(centroid_coordinates)

# Visualize the clustering
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=clustering.labels_, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='*', s=200, edgecolor='black')
plt.title('Hierarchical Agglomerative Clustering')
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