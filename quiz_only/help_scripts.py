import pandas as pd, matplotlib.pyplot as plt
s = pd.read_csv("outputs/famd_scores_with_clusters.csv")
plt.figure(figsize=(7,5))
plt.scatter(s["FAMD_1"], s["FAMD_2"], s=8, alpha=0.6, c=s["cluster"])
plt.xlabel("FAMD 1"); plt.ylabel("FAMD 2"); plt.title("FAMD (colored by cluster)")
plt.tight_layout(); plt.show()
