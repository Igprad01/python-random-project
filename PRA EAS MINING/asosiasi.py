import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Data transaksi
transactions = [
    ['Sabun', 'Sampo', 'Sikat gigi', 'Pasta gigi'],
    ['Sabun', 'Sikat gigi', 'Sampo'],
    ['Sampo', 'Sikat gigi', 'Pasta gigi'],
    ['Pasta gigi', 'Sampo'],
    ['Sampo', 'Pasta gigi']
]

# Mengubah data transaksi menjadi DataFrame
all_items = sorted(set(item for transaction in transactions for item in transaction))
encoded_vals = []
for transaction in transactions:
    row = {item: (item in transaction) for item in all_items}
    encoded_vals.append(row)
df = pd.DataFrame(encoded_vals)

# Menjalankan algoritma Apriori
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)

# Menghasilkan aturan asosiasi
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules)