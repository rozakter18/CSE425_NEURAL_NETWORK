# -*- coding: utf-8 -*-
"""
Easy Clustering Script

This script performs clustering on latent representations from VAE and PCA,
and generates visualizations using t-SNE and UMAP.

Usage:
1. Ensure latent files exist in the processed directory.
2. Run: python easy_clustering.py
3. Results will be saved to the easy_vae/ and easy_pca/ directories.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import umap.umap_ as UMAP
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Local paths
BASE_DIR = "C:/Users/YamalxMessi/Desktop/Project"
PATH_LATENT_VAE = f"{BASE_DIR}/processed/latent_vae_.npy"
PATH_PCA = f"{BASE_DIR}/processed/latent_pca.npy"
PATH_META = f"{BASE_DIR}/processed/processed_metadata.csv"

Z_vae = np.load(PATH_LATENT_VAE)
print("VAE latent:", Z_vae.shape)

try:
    Z_pca = np.load(PATH_PCA)
    print("PCA:", Z_pca.shape)
except FileNotFoundError:
    Z_pca = None
    print("No PCA baseline found.")

meta = pd.read_csv(PATH_META)
print("Meta:", meta.shape)

min_len = min(len(Z_vae), len(meta))

Z_vae = Z_vae[:min_len]
meta = meta.iloc[:min_len].reset_index(drop=True)

if Z_pca is not None:
    Z_pca = Z_pca[:min_len]

print("After fix:Z:", Z_vae.shape, "Meta:", meta.shape)

def improved_elbow(X, k_min=2, k_max=12):
    """Compute improved elbow method for optimal k selection."""
    inertias = []
    Ks = range(k_min, k_max + 1)

    for k in Ks:
        km = KMeans(n_clusters=k, random_state=0)
        km.fit(X)
        inertias.append(km.inertia_)

    d2 = []
    for i in range(1, len(inertias) - 1):
        d2.append(inertias[i-1] - 2*inertias[i] + inertias[i+1])

    best_k = Ks[np.argmax(d2) + 1]
    return best_k, list(Ks), inertias


def run_kmeans_and_plots(X, method_name, out_prefix):
    """Run K-means clustering and generate plots for given features."""
    print(f"Clustering with {method_name} features")

    best_k, Ks, inertias = improved_elbow(X)
    print(f"Best K (improved elbow) = {best_k}")

    plt.figure()
    plt.plot(Ks, inertias, marker="o")
    plt.xlabel("K")
    plt.ylabel("Inertia")
    plt.title(f"Inertia vs K — {method_name}")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{out_prefix}_elbow.png", dpi=200)
    plt.close()

    km = KMeans(n_clusters=best_k, random_state=0)
    labels = km.fit_predict(X)

    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    print(f"Silhouette: {sil:.4f}")
    print(f"Calinski–Harabasz: {ch:.2f}")

    z_tsne = TSNE(n_components=2, perplexity=30, learning_rate=200).fit_transform(X)

    plt.figure(figsize=(6, 6))
    plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=labels, cmap="tab20", s=8)
    plt.title(f"t-SNE — {method_name} (K={best_k})")
    plt.xticks([]); plt.yticks([])
    plt.savefig(f"{out_prefix}_tsne.png", dpi=200)
    plt.close()

    reducer = UMAP.UMAP(n_components=2, random_state=0)
    z_umap = reducer.fit_transform(X)

    plt.figure(figsize=(6, 6))
    plt.scatter(z_umap[:, 0], z_umap[:, 1], c=labels, cmap="tab20", s=8)
    plt.title(f"UMAP — {method_name} (K={best_k})")
    plt.xticks([]); plt.yticks([])
    plt.savefig(f"{out_prefix}_umap.png", dpi=200)
    plt.close()

    out_meta = meta.copy()
    out_meta[f"cluster_{method_name.lower()}"] = labels
    out_meta.to_csv(f"{out_prefix}_clusters.csv", index=False)

    metrics = {
        "method": method_name,
        "best_k": best_k,
        "silhouette": sil,
        "calinski_harabasz": ch,
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f"{out_prefix}_metrics.csv", index=False)

    print(f"Saved: {out_prefix}_clusters.csv, {out_prefix}_metrics.csv")
    print(f"Plots: {out_prefix}_elbow.png, {out_prefix}_tsne.png, {out_prefix}_umap.png")

    return labels, metrics

"""# **EASY TASK**"""

labels_vae, metrics_vae = run_kmeans_and_plots(
    X=Z_vae,
    method_name="VAE",
    out_prefix=f"{BASE_DIR}/easy_vae"
)

if Z_pca is not None:
    labels_pca, metrics_pca = run_kmeans_and_plots(
        X=Z_pca,
        method_name="PCA",
        out_prefix=f"{BASE_DIR}/easy_pca"
    )
else:
    print("\nNo PCA features file found")

