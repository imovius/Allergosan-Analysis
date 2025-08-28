import pandas as pd
df = pd.read_csv("outputs/B_noLTV_labels_pruned.csv")
print(df['cluster_final'].value_counts(normalize=True))
