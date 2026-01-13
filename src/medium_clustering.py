# -*- coding: utf-8 -*-
"""
Medium Clustering Script

This script performs comprehensive clustering evaluation on VAE and PCA latents
using multiple algorithms and metrics.

Usage:
1. Ensure latent files exist from medium_vae.py.
2. Run: python medium_clustering.py
3. Results will be saved to results directory.
"""

import os
import json
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

# Local paths
BASE_DIR = "C:/Users/YamalxMessi/Desktop/Project"
RESULTS_DIR = os.path.join(BASE_DIR, "results")

Z_VAE_PATH = os.path.join(RESULTS_DIR, "Z_medium.npy")
Z_PCA_PATH = os.path.join(RESULTS_DIR, "Z_pca_medium.npy")

OUT_METRICS = os.path.join(RESULTS_DIR, "clustering_medium_metrics.csv")
OUT_DETAILS = os.path.join(RESULTS_DIR, "clustering_medium_details.json")

MANUAL_K_LIST = [5, 10, 15, 20]

def load_latents():
    """Load VAE and PCA latents and sanitize NaNs/Infs."""
    Z_vae = np.load(Z_VAE_PATH)
    Z_pca = np.load(Z_PCA_PATH)

    print("VAE latent shape:", Z_vae.shape)
    print("PCA latent shape:", Z_pca.shape)

    # replace NaN / Inf with large finite numbers
    Z_vae = np.nan_to_num(Z_vae, nan=0.0, posinf=1e6, neginf=-1e6)
    Z_pca = np.nan_to_num(Z_pca, nan=0.0, posinf=1e6, neginf=-1e6)

    return Z_vae, Z_pca

def compute_elbow_k(X, k_min=2, k_max=20, random_state=42):
    """Compute optimal K using elbow method."""
    inertias = []
    ks = list(range(k_min, k_max + 1))

    print(f"[auto_k_elbow] Scanning K from {k_min} to {k_max} ...")
    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        km.fit(X)
        inertias.append(km.inertia_)

    inertias = np.array(inertias, dtype=float)


    x1, y1 = ks[0], inertias[0]
    x2, y2 = ks[-1], inertias[-1]


    distances = []
    for k, sse in zip(ks, inertias):
        num = abs((y2 - y1) * k - (x2 - x1) * sse + x2 * y1 - y2 * x1)
        den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        distances.append(num / den)

    distances = np.array(distances)
    best_idx = np.argmax(distances)
    best_k = ks[best_idx]

    print("[auto_k_elbow] Inertias:", dict(zip(ks, inertias)))
    print("[auto_k_elbow] Distances:", dict(zip(ks, distances)))
    print(f"[auto_k_elbow] Best K by elbow = {best_k}")

    info = {
        "ks": ks,
        "inertias": inertias.tolist(),
        "distances": distances.tolist(),
        "best_k": int(best_k),
    }
    return best_k, info

def evaluate_clustering(X, labels, algo, rep, k=None, mode="auto", extra=None):

    labels = np.asarray(labels)
    unique = np.unique(labels)
    n_clusters = len(unique)
    if -1 in unique:
        n_clusters_eff = n_clusters - 1
    else:
        n_clusters_eff = n_clusters

    valid = n_clusters_eff >= 2 and len(labels) > n_clusters_eff

    if valid:
        try:
            sil = silhouette_score(X, labels)
            db = davies_bouldin_score(X, labels)
            ch = calinski_harabasz_score(X, labels)
        except Exception as e:
            print(f"[WARN] Metric error for {algo}-{rep}-K={k}: {e}")
            sil = np.nan
            db = np.nan
            ch = np.nan
    else:
        sil = np.nan
        db = np.nan
        ch = np.nan

    return {
        "representation": rep,
        "algorithm": algo,
        "mode": mode,
        "k": k,
        "num_clusters": int(n_clusters_eff),
        "silhouette": float(sil) if not np.isnan(sil) else np.nan,
        "davies_bouldin": float(db) if not np.isnan(db) else np.nan,
        "calinski_harabasz": float(ch) if not np.isnan(ch) else np.nan,
        "extra": extra,
    }

def run_kmeans_block(X, rep_name, results, manual_ks, random_state=42):

    best_k, elbow_info = compute_elbow_k(X, k_min=2, k_max=20, random_state=random_state)
    km_auto = KMeans(n_clusters=best_k, random_state=random_state, n_init="auto")
    labels_auto = km_auto.fit_predict(X)

    res_auto = evaluate_clustering(
        X, labels_auto,
        algo="kmeans", rep=rep_name,
        k=best_k, mode="auto",
        extra={"elbow_info": elbow_info}
    )
    results.append(res_auto)

    for k in manual_ks:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = km.fit_predict(X)
        res = evaluate_clustering(
            X, labels,
            algo="kmeans", rep=rep_name,
            k=k, mode="manual",
            extra=None
        )
        results.append(res)

def run_agglo_block(X, rep_name, results, k_for_auto, manual_ks):

    agg_auto = AgglomerativeClustering(n_clusters=k_for_auto, linkage="ward")
    labels_auto = agg_auto.fit_predict(X)

    res_auto = evaluate_clustering(
        X, labels_auto,
        algo="agglomerative", rep=rep_name,
        k=k_for_auto, mode="auto",
        extra={"linkage": "ward"}
    )
    results.append(res_auto)

    for k in manual_ks:
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = agg.fit_predict(X)
        res = evaluate_clustering(
            X, labels,
            algo="agglomerative", rep=rep_name,
            k=k, mode="manual",
            extra={"linkage": "ward"}
        )
        results.append(res)

def run_dbscan_block(X, rep_name, results):

    eps_list = [0.5, 1.0, 1.5, 2.0]
    min_samples_list = [5, 10]

    best_cfg = None
    best_sil = -np.inf
    best_res = None

    for eps in eps_list:
        for m in min_samples_list:
            db = DBSCAN(eps=eps, min_samples=m)
            labels = db.fit_predict(X)

            res = evaluate_clustering(
                X, labels,
                algo="dbscan", rep=rep_name,
                k=None, mode="grid",
                extra={"eps": eps, "min_samples": m}
            )


            sil = res["silhouette"]
            if np.isnan(sil):
                continue

            if sil > best_sil:
                best_sil = sil
                best_cfg = (eps, m)
                best_res = res

    if best_res is not None:
        print(f"[DBSCAN-{rep_name}] Best eps={best_cfg[0]}, min_samples={best_cfg[1]}, silhouette={best_sil:.3f}")
        results.append(best_res)
    else:

        print(f"[DBSCAN-{rep_name}] No valid clustering found (all degenerate).")
        results.append({
            "representation": rep_name,
            "algorithm": "dbscan",
            "mode": "grid",
            "k": None,
            "num_clusters": 0,
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
            "extra": {"note": "no valid clusterings (all single/noise)"},
        })

def main():
    Z_vae, Z_pca = load_latents()

    all_results = []
    extra_info = {"vae": {}, "pca": {}}

    print("\nClustering with VAE latent")
    k_auto_vae, elbow_vae = compute_elbow_k(Z_vae, k_min=2, k_max=20)
    extra_info["vae"]["auto_k"] = elbow_vae


    km = KMeans(n_clusters=k_auto_vae, random_state=42, n_init="auto")
    labels_km_auto = km.fit_predict(Z_vae)
    all_results.append(
        evaluate_clustering(
            Z_vae, labels_km_auto,
            algo="kmeans", rep="VAE",
            k=k_auto_vae, mode="auto",
            extra={"elbow_info": elbow_vae}
        )
    )


    for k in MANUAL_K_LIST:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(Z_vae)
        all_results.append(
            evaluate_clustering(Z_vae, labels,
                                algo="kmeans", rep="VAE",
                                k=k, mode="manual")
        )


    agg_auto = AgglomerativeClustering(n_clusters=k_auto_vae, linkage="ward")
    labels_agg_auto = agg_auto.fit_predict(Z_vae)
    all_results.append(
        evaluate_clustering(Z_vae, labels_agg_auto,
                            algo="agglomerative", rep="VAE",
                            k=k_auto_vae, mode="auto",
                            extra={"linkage": "ward"})
    )

    for k in MANUAL_K_LIST:
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = agg.fit_predict(Z_vae)
        all_results.append(
            evaluate_clustering(Z_vae, labels,
                                algo="agglomerative", rep="VAE",
                                k=k, mode="manual",
                                extra={"linkage": "ward"})
        )


    run_dbscan_block(Z_vae, "VAE", all_results)


    print("\nClustering with PCA latent")
    k_auto_pca, elbow_pca = compute_elbow_k(Z_pca, k_min=2, k_max=20)
    extra_info["pca"]["auto_k"] = elbow_pca


    km = KMeans(n_clusters=k_auto_pca, random_state=42, n_init="auto")
    labels_km_auto = km.fit_predict(Z_pca)
    all_results.append(
        evaluate_clustering(
            Z_pca, labels_km_auto,
            algo="kmeans", rep="PCA",
            k=k_auto_pca, mode="auto",
            extra={"elbow_info": elbow_pca}
        )
    )

    for k in MANUAL_K_LIST:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(Z_pca)
        all_results.append(
            evaluate_clustering(Z_pca, labels,
                                algo="kmeans", rep="PCA",
                                k=k, mode="manual")
        )


    agg_auto = AgglomerativeClustering(n_clusters=k_auto_pca, linkage="ward")
    labels_agg_auto = agg_auto.fit_predict(Z_pca)
    all_results.append(
        evaluate_clustering(Z_pca, labels_agg_auto,
                            algo="agglomerative", rep="PCA",
                            k=k_auto_pca, mode="auto",
                            extra={"linkage": "ward"})
    )

    for k in MANUAL_K_LIST:
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = agg.fit_predict(Z_pca)
        all_results.append(
            evaluate_clustering(Z_pca, labels,
                                algo="agglomerative", rep="PCA",
                                k=k, mode="manual",
                                extra={"linkage": "ward"})
        )


    run_dbscan_block(Z_pca, "PCA", all_results)

    # ---------- Save results ----------
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(OUT_METRICS, index=False)
    print("\nSaved clustering metrics to:", OUT_METRICS)
    print(df_results.head())

    with open(OUT_DETAILS, "w") as f:
        json.dump(extra_info, f, indent=2)
    print("Saved extra info (auto-K details) to:", OUT_DETAILS)


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt

RESULTS_DIR = "/content/drive/MyDrive/DATASET(CSE425)/results"

df = pd.read_csv(f"{RESULTS_DIR}/clustering_medium_metrics.csv")
Z_vae = np.load(f"{RESULTS_DIR}/Z_medium.npy")
Z_pca = np.load(f"{RESULTS_DIR}/Z_pca_medium.npy")

def plot_latent(X, labels, title):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # TSNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
    Zt = tsne.fit_transform(X)
    axs[0].scatter(Zt[:,0], Zt[:,1], s=5, c=labels, cmap='tab20')
    axs[0].set_title(f"{title} — TSNE")

    # UMAP
    um = umap.UMAP(n_neighbors=20, min_dist=0.05)
    Zu = um.fit_transform(X)
    axs[1].scatter(Zu[:,0], Zu[:,1], s=5, c=labels, cmap='tab20')
    axs[1].set_title(f"{title} — UMAP")

    plt.show()

best_k_vae = df[(df["representation"]=="VAE") & (df["mode"]=="auto")]["k"].values[0]
best_k_pca = df[(df["representation"]=="PCA") & (df["mode"]=="auto")]["k"].values[0]

print(f"Best K VAE: {best_k_vae}")
print(f"Best K PCA: {best_k_pca}")

km = KMeans(n_clusters=int(best_k_vae), n_init='auto', random_state=42)
labels = km.fit_predict(Z_vae)
plot_latent(Z_vae, labels, f"VAE — Best K={best_k_vae}")

km = KMeans(n_clusters=int(best_k_pca), n_init='auto', random_state=42)
labels = km.fit_predict(Z_pca)
plot_latent(Z_pca, labels, f"PCA — Best K={best_k_pca}")

for k in [5, 10, 15, 20]:
    for rep, Z in [("VAE", Z_vae), ("PCA", Z_pca)]:
        km = KMeans(n_clusters=k, n_init='auto', random_state=42)
        labels = km.fit_predict(Z)
        plot_latent(Z, labels, f"{rep} — Manual K={k}")

agg_vae = AgglomerativeClustering(n_clusters=int(best_k_vae), linkage='ward')
labels_agg_vae = agg_vae.fit_predict(Z_vae)
plot_latent(Z_vae, labels_agg_vae, f"VAE — Agglomerative (K={best_k_vae})")


agg_pca = AgglomerativeClustering(n_clusters=int(best_k_pca), linkage='ward')
labels_agg_pca = agg_pca.fit_predict(Z_pca)
plot_latent(Z_pca, labels_agg_pca, f"PCA — Agglomerative (K={best_k_pca})")

from sklearn.preprocessing import StandardScaler

Z_vae_scaled = StandardScaler().fit_transform(Z_vae)
Z_pca_scaled = StandardScaler().fit_transform(Z_pca)


db_vae = DBSCAN(eps=1.5, min_samples=10).fit(Z_vae_scaled)
labels_db_vae = db_vae.labels_
plot_latent(Z_vae, labels_db_vae, f"VAE — DBSCAN (eps=1.5, min=10)")


db_pca = DBSCAN(eps=1.8, min_samples=10).fit(Z_pca_scaled)
labels_db_pca = db_pca.labels_
plot_latent(Z_pca, labels_db_pca, f"PCA — DBSCAN (eps=1.8, min=10)")

def show_cluster_stats(name, labels):
    unique, counts = np.unique(labels, return_counts=True)
    print(f"{name} cluster distribution:")
    for u,c in zip(unique, counts):
        print(f"  {u}: {c}")
    print()

show_cluster_stats("VAE-AGG", labels_agg_vae)
show_cluster_stats("PCA-AGG", labels_agg_pca)
show_cluster_stats("VAE-DBSCAN", labels_db_vae)
show_cluster_stats("PCA-DBSCAN", labels_db_pca)