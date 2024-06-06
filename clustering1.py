# Membangkitkan data:
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

# Generate data
data = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=1.6, random_state=50)
points = data[0]

# Plot data yang dihasilkan
plt.scatter(data[0][:, 0], data[0][:, 1], c=data[1], cmap='viridis')
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.show()

# Menggunakan KMeans untuk mengelompokkan data menjadi 4 cluster
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(points)
print(kmeans.cluster_centers_)
y_km = kmeans.fit_predict(points)

# Menampilkan hasil clustering yang baru
plt.scatter(points[y_km == 0, 0], points[y_km == 0, 1], s=500, c='blue')
plt.scatter(points[y_km == 1, 0], points[y_km == 1, 1], s=500, c='black')
plt.scatter(points[y_km == 2, 0], points[y_km == 2, 1], s=500, c='red')
plt.scatter(points[y_km == 3, 0], points[y_km == 3, 1], s=500, c='cyan')

# Opsional: menampilkan pusat cluster
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', marker='X')