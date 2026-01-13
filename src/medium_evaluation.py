# -*- coding: utf-8 -*-
"""
Medium Evaluation Script

This script evaluates clustering results from medium_clustering.py and generates summary statistics.

Usage:
1. Run medium_clustering.py first.
2. Run: python medium_evaluation.py
3. Summary statistics will be printed.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Local path
BASE_DIR = "C:/Users/YamalxMessi/Desktop/Project"
CSV_PATH = os.path.join(BASE_DIR, "results/clustering_medium_metrics.csv")

df = pd.read_csv(CSV_PATH)
print(df.head())

best_sil = (
    df.sort_values(by="silhouette", ascending=False)
      .groupby(["representation", "algorithm"])
      .first()[["silhouette", "k", "num_clusters"]]
)

print("\nBest Silhouette per representation+algorithm")
print(best_sil)

best_ch = (
    df.sort_values(by="calinski_harabasz", ascending=False)
      .groupby(["representation", "algorithm"])
      .first()[["calinski_harabasz", "k"]]
)

print("\nBest Calinski-Harabasz")
print(best_ch)

best_db = (
    df.sort_values(by="davies_bouldin", ascending=True)
      .groupby(["representation", "algorithm"])
      .first()[["davies_bouldin", "k"]]
)

print("\nBest Davies-Bouldin")
print(best_db)

summary = df.groupby("representation").agg({
    "silhouette":"mean",
    "calinski_harabasz":"mean",
    "davies_bouldin":"mean",
    "num_clusters":"mean"
})

print("\nRepresentation Summary (mean)")
print(summary)

winner = {
    "best_silhouette": best_sil.sort_values("silhouette", ascending=False).head(1),
    "best_CH": best_ch.sort_values("calinski_harabasz", ascending=False).head(1),
    "best_DB": best_db.sort_values("davies_bouldin", ascending=True).head(1),
}

for k,v in winner.items():
    print(f"\n{k.upper()}")
    print(v)

import matplotlib.pyplot as plt
import numpy as np

df_grp = df.groupby("representation").agg({
    "silhouette":"mean",
    "calinski_harabasz":"mean",
    "davies_bouldin":"mean"
}).reset_index()


df_grp["inv_db"] = df_grp["davies_bouldin"].max() - df_grp["davies_bouldin"]

labels = ["Silhouette", "Calinski-Harabasz", "Inverse DB"]
metrics = ["silhouette", "calinski_harabasz", "inv_db"]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8,5))

vae_vals = df_grp[df_grp["representation"]=="VAE"][metrics].values.flatten()
pca_vals = df_grp[df_grp["representation"]=="PCA"][metrics].values.flatten()

bar1 = ax.bar(x - width/2, vae_vals, width, label="VAE", alpha=0.85)
bar2 = ax.bar(x + width/2, pca_vals, width, label="PCA", alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_title("Clustering Metric Comparison — VAE vs PCA")
ax.set_ylabel("Relative Score (higher = better)")
ax.legend()

for bars in [bar1, bar2]:
    for b in bars:
        height = b.get_height()
        ax.annotate(f"{height:.3f}", xy=(b.get_x()+b.get_width()/2, height),
                    xytext=(0,3), textcoords="offset points", ha='center', fontsize=9)

plt.show()

rel = (df_grp[df_grp.representation=="VAE"][["silhouette","calinski_harabasz"]].values /
       df_grp[df_grp.representation=="PCA"][["silhouette","calinski_harabasz"]].values)*100

labels = ["Silhouette","CH Index"]
plt.bar(labels, rel.flatten())
plt.title("Relative Improvement of VAE over PCA (%)")
plt.ylabel("% improvement")
plt.show()

import seaborn as sns

df_z = df_grp.copy()
df_z[["silhouette","calinski_harabasz","davies_bouldin"]] = \
    (df_z[["silhouette","calinski_harabasz","davies_bouldin"]] -
     df_z[["silhouette","calinski_harabasz","davies_bouldin"]].mean()) / \
     df_z[["silhouette","calinski_harabasz","davies_bouldin"]].std()

plt.figure(figsize=(5,3))
sns.heatmap(df_z.set_index("representation"), annot=True, cmap="coolwarm")
plt.title("Z-score Normalized Comparison")
plt.show()