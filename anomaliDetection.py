import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
# Generate synthetic data
np.random.seed(42)
data = np.random.randn(100, 2)  # 100 samples, 2 features
data = np.append(data, [[10, 10], [15, 15]], axis=0)  # Add anomalies
# Convert to DataFrame
df = pd.DataFrame(data, columns=['Feature1', 'Feature2'])
# Visualize the data
plt.scatter(df['Feature1'], df['Feature2'])
plt.title('Data Visualization')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.show()
# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
# Train the Isolation Forest model
model = IsolationForest(contamination=0.05)  # Assuming 5% of data are anomalies
model.fit(df_scaled)
# Predict anomalies
df['Anomaly'] = model.predict(df_scaled)
df['Anomaly'] = df['Anomaly'].map({1: 0, -1: 1})  # 0: normal, 1: anomaly
# Plot the anomalies
plt.scatter(df['Feature1'], df['Feature2'], c=df['Anomaly'], cmap='coolwarm')
plt.title('Anomaly Detection')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.show()
# Print the anomalies
anomalies = df[df['Anomaly'] == 1]
print("Anomalies found:\n", anomalies)