import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Dataset
data = np.array([
    [2, 5],
    [4, 2],
    [3, 3],
    [3, 4],
    [5, 7],
    [6, 6],
    [7, 5]
])

# Tentukan jumlah cluster
n_clusters = 2

# Latih model K-means
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(data)

# Dapatkan hasil cluster
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Visualisasi hasil clustering
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', label='Centroids')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('K-means Clustering')
plt.legend()
plt.show()

# Menampilkan hasil clustering
for i, point in enumerate(data):
    print(f"Point {point}, Cluster: {labels[i]}")
