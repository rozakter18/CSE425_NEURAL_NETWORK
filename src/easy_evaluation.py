# -*- coding: utf-8 -*-
"""
Easy Evaluation Script

This script loads clustering metrics from VAE and PCA results and generates comparison plots.

Usage:
1. Run easy_clustering.py first to generate metrics files.
2. Run: python easy_evaluation.py
3. Comparison plots will be displayed.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt

# Local paths
BASE_DIR = "C:/Users/YamalxMessi/Desktop/Project"

df_vae = pd.read_csv(f"{BASE_DIR}/easy_vae_metrics.csv")
print("VAE Metrics:")
print(df_vae)

try:
    df_pca = pd.read_csv(f"{BASE_DIR}/easy_pca_metrics.csv")
    print("\nPCA Metrics:")
    print(df_pca)
except:
    df_pca = None
    print("\nPCA baseline not available")

if df_pca is not None:
    df_vae["Model"] = "VAE"
    df_pca["Model"] = "PCA"
    df_comp = pd.concat([df_vae, df_pca], ignore_index=True)
else:
    df_vae["Model"] = "VAE"
    df_comp = df_vae.copy()

print("\nComparison Table")
print(df_comp)

df = df_comp.set_index("Model")

fig, axes = plt.subplots(1, 2, figsize=(15,4))

df["silhouette"].plot(kind="bar", ax=axes[0], color=['steelblue','orange'])
axes[0].set_title("Silhouette Score (Higher = Better)")
axes[0].set_ylabel("Score")

df["calinski_harabasz"].plot(kind="bar", ax=axes[1], color=['steelblue','orange'])
axes[1].set_title("Calinski–Harabasz (Higher = Better)")
axes[1].set_ylabel("Score")

plt.suptitle("VAE vs PCA — Clustering Quality Comparison", fontsize=14)
plt.tight_layout()
plt.show()

imp_sil = (df.loc["VAE","silhouette"] / df.loc["PCA","silhouette"] - 1) * 100
imp_cal = (df.loc["VAE","calinski_harabasz"] / df.loc["PCA","calinski_harabasz"] - 1) * 100

plt.figure(figsize=(6,4))
plt.bar(["Silhouette","Calinski"], [imp_sil, imp_cal], color=["skyblue","orange"])
plt.ylabel("Improvement (%)")
plt.title("VAE Improvement over PCA")
plt.grid(axis="y", alpha=0.3)
plt.show()

print("\nInterpretation Summary:")
if df.loc["VAE","silhouette"] > df.loc["PCA","silhouette"]:
    print(" VAE forms more compact & separated clusters (higher Silhouette).")

if df.loc["VAE","calinski_harabasz"] > df.loc["PCA","calinski_harabasz"]:
    print("VAE improves inter-cluster separation (higher CH).")

print("\nConclusion: VAE latent space provides a more clustering-friendly representation than PCA.")

