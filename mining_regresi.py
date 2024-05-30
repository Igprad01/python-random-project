import numpy as np

# Data penjualan triwulan
penjualan = np.array([11, 23, 30, 45])
# Bulan 1 hingga 4
bulan = np.array([1, 2, 3, 4])

# Hitung nilai rata-rata
mean_x = np.mean(bulan)
mean_y = np.mean(penjualan)

# Hitung koefisien q1 (slope)
q1_numerator = np.sum((bulan - mean_x) * (penjualan - mean_y))
q1_denominator = np.sum((bulan - mean_x) ** 2)
q1 = q1_numerator / q1_denominator

# Hitung koefisien q0 (intercept)
q0 = mean_y - q1 * mean_x

# Prediksi penjualan untuk bulan ke-5
bulan_ke_5 = 5
prediksi_penjualan_bulan_5 = q0 + q1 * bulan_ke_5

print(f"q0 (intercept): {q0:.2f}")
print(f"q1 (slope): {q1:.2f}")
print(f"Prediksi penjualan untuk bulan ke-5: {prediksi_penjualan_bulan_5:.2f}")