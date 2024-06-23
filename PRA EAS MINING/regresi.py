import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset
X = np.array([3, 2, 5, 9, 15]).reshape(-1, 1)  # Koordinat x
y = np.array([4, 3, 7, 12, 20])  # Koordinat y

# Latih model regresi linear
model = LinearRegression()
model.fit(X, y)

# Prediksi menggunakan model regresi
y_pred = model.predict(X)

# Menampilkan hasil regresi
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, y_pred, color='red', label='Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# Klasifikasi berdasarkan ambang batas
# Misalnya, kita ingin mengklasifikasikan apakah y > 10
threshold = 10
classification = y_pred > threshold

# Menampilkan hasil klasifikasi
for i, point in enumerate(X):
    print(f"Point {point[0]}, Prediction: {y_pred[i]:.2f}, Classification: {'Above Threshold' if classification[i] else 'Below Threshold'}")
