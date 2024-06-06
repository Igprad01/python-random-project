# %matplotlib inline
from copy import deepcopy
import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# Import data
data = pd.read_csv('https://github.com/mubaris/friendly-fortnight/raw/master/xclara.csv')
print(data.shape)
print(data.head())

# Menampilkan sebaran data
f1 = data['V1'].values
f2 = data['V2'].values
x = np.array(list(zip(f1, f2)))

# KMeans clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(x)

# Mendapatkan centroid
centroids = kmeans.cluster_centers_
print("Centroids:\n", centroids)

# Plot sebaran data dan centroid
plt.scatter(f1, f2, c='black', s=7, label='Data Points')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='X', label='Centroids')
plt.xlabel('V1')
plt.ylabel('V2')
plt.title('Data Scatter Plot with KMeans Centroids')
plt.legend()
plt.show()
