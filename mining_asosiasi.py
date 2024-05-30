import pycaret

# menggunakan mlxtend untuk analisis asosiasi
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load data
from pycaret.datasets import get_data
data = get_data('france')
print(data.head())

# Prepare data for mlxtend
basket = (data
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

# Convert quantities to 1 or 0
basket_sets = basket.applymap(lambda x: 1 if x >= 1 else 0)

# Apply apriori
frequent_itemsets = apriori(basket_sets, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
print(rules)

# Plotting (2D and 3D)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 2D plot
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Support vs Confidence')
plt.show()

# 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(rules['support'], rules['confidence'], rules['lift'], alpha=0.5)
ax.set_xlabel('Support')
ax.set_ylabel('Confidence')
ax.set_zlabel('Lift')
plt.title('Support vs Confidence vs Lift')
plt.show()