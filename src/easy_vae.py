# -*- coding: utf-8 -*-
"""
Easy VAE Training Script

This script trains a simple VAE on audio statistics features and saves latent representations.

Usage:
1. Ensure audio stats chunks and metadata are available.
2. Run: python easy_vae.py
3. Latent representations will be saved to processed directory.
"""

import glob
import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Local paths
BASE_DIR = "C:/Users/YamalxMessi/Desktop/Project"
STATS_DIR = os.path.join(BASE_DIR, "processed/audio_stats_chunks")
META_PATH = os.path.join(BASE_DIR, "processed/processed_metadata.csv")

OUT_LATENT_VAE = os.path.join(BASE_DIR, "processed/latent_vae_.npy")
OUT_PCA_FEATS = os.path.join(BASE_DIR, "processed/latent_pca.npy")
OUT_PCA_MODEL = os.path.join(BASE_DIR, "processed/pca_model.pkl")

BATCH_SIZE = 64
LATENT_DIM = 16
EPOCHS = 80
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

meta = pd.read_csv(META_PATH)
meta = meta[['TrackID', 'sample_rate', 'Duration_sec']]
meta['Duration_sec'] = meta['Duration_sec'].astype(float)

def load_stats_and_match(STATS_DIR: str, meta: pd.DataFrame):
    """Load audio stats chunks and match with metadata."""
    if os.path.basename(STATS_DIR) != "audio_stats_chunks":
        STATS_DIR = os.path.join(STATS_DIR, "audio_stats_chunks")

    stats_files = sorted(glob.glob(os.path.join(STATS_DIR, "stats_chunk_*.npy")))
    ids_files   = sorted(glob.glob(os.path.join(STATS_DIR, "ids_chunk_*.npy")))

    print(f"Found {len(stats_files)} stats chunks, {len(ids_files)} ids chunks")

    assert len(stats_files) == len(ids_files), "stats / ids chunk count mismatch"

    X_list, id_list = [], []

    for sfile, ifile in tqdm(zip(stats_files, ids_files),
                             total=len(stats_files),
                             desc="Loading audio stats"):
        X   = np.load(sfile, allow_pickle=True)
        ids = np.load(ifile, allow_pickle=True).tolist()

        n = min(len(X), len(ids))
        if n == 0:
            continue

        X_list.append(X[:n])
        id_list.extend(ids[:n])

    if not X_list:
        raise RuntimeError("No rows loaded – check STATS_DIR/path patterns.")

    X_all  = np.vstack(X_list)
    id_list = [str(x) for x in id_list]
    print("Raw stacked shape:", X_all.shape, " IDs:", len(id_list))

    meta = meta.set_index("TrackID")
    mask = [tid in meta.index for tid in id_list]
    valid_ids = [tid for tid, keep in zip(id_list, mask) if keep]

    X_filtered = X_all[mask]
    meta_filtered = meta.loc[valid_ids].reset_index()

    print("Final aligned:", X_filtered.shape, meta_filtered.shape)
    return X_filtered, meta_filtered


X_stats, meta_aligned = load_stats_and_match(STATS_DIR, meta)
print("Stats loaded:", X_stats.shape)


aux = meta_aligned[["sample_rate", "Duration_sec"]].values.astype(float)
X = np.hstack([X_stats, aux])
print("Final input feature shape:", X.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

"""# **EASY TASK**"""

class VAE(nn.Module):
    """Simple Variational Autoencoder."""

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        recon = self.dec(z)
        return recon, mu, logvar

def vae_loss(recon, x, mu, logvar):
    """Compute VAE loss (reconstruction + KL)."""
    recon_loss = nn.MSELoss()(recon, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl

class EarlyStopper:
    """Early stopping utility."""

    def __init__(self, patience=5, min_delta=1e-4):
        self.best = None
        self.counter = 0
        self.patience = patience
        self.min_delta = min_delta

    def step(self, val):
        if self.best is None or val < self.best - self.min_delta:
            self.best = val
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

input_dim = X_scaled.shape[1]
vae = VAE(input_dim, LATENT_DIM).to(DEVICE)
opt = torch.optim.Adam(vae.parameters(), lr=LR)
stopper = EarlyStopper()

X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)

for epoch in range(EPOCHS):
    opt.zero_grad()
    recon, mu, logvar = vae(X_tensor)
    loss = vae_loss(recon, X_tensor, mu, logvar)
    loss.backward()
    opt.step()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss = {loss.item():.4f}")

    if stopper.step(loss.item()):
        print("Early stopping triggered!")
        break

with torch.no_grad():
    _, Z, _ = vae(X_tensor)
Z = Z.cpu().numpy()
np.save(OUT_LATENT_VAE, Z)
print("Saved latent VAE:", Z.shape)

pca = PCA(n_components=LATENT_DIM)
Zpca = pca.fit_transform(X_scaled)

np.save(OUT_PCA_FEATS, Zpca)
joblib.dump(pca, OUT_PCA_MODEL)

print("Saved PCA:", Zpca.shape)
