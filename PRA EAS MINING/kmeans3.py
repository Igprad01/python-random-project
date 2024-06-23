from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Data tunggal
data = np.array([3, 5, 8, 9, 15, 17, 18, 19]).reshape(-1, 1)

# Inisialisasi model KMeans dengan 2 cluster
kmeans = KMeans(n_clusters=2, random_state=0)

# Fit model ke data
kmeans.fit(data)

# Mendapatkan label cluster untuk setiap data point
labels = kmeans.labels_

# Mendapatkan centroids dari setiap cluster
centroids = kmeans.cluster_centers_

# Plot hasil clustering
plt.scatter(data, np.zeros_like(data), c=labels, s=100, cmap='viridis')
plt.scatter(centroids, np.zeros_like(centroids), c='red', s=300, alpha=0.5)
plt.title('K-means Clustering of Data')
plt.xlabel('Data Points')
plt.yticks([])
plt.show()

# Menampilkan hasil
print("Data Points: ", data.flatten())
print("Cluster Labels: ", labels)
print("Centroids: ", centroids.flatten())
