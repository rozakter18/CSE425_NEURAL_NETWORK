# -*- coding: utf-8 -*-
"""
Multimodal Music Analysis Script

This script performs comprehensive multimodal music analysis using various autoencoder models
(Beta-VAE, CVAE, Autoencoder) and compares them with PCA for clustering tasks.

Features:
- Trains lightweight models on mel spectrograms, lyrics embeddings, and tabular features
- Evaluates clustering quality using multiple metrics
- Generates visualizations of latent spaces and reconstructions
- Includes disentanglement analysis for VAE models

Requirements:
- Python 3.8+
- PyTorch, scikit-learn, pandas, numpy, matplotlib, tqdm, umap-learn

Usage:
1. Ensure data files are in the specified directories (see CONFIG)
2. Run: python hard_task.py
3. Results will be saved to the OUT_DIR

Data Requirements:
- Processed metadata CSV
- Mel spectrograms in NPZ format
- Lyrics embeddings in NPY format
- Audio statistics CSV
"""

import gc
import logging
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Scikit-learn version: {sklearn.__version__}")

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = "C:/Users/YamalxMessi/Desktop/Project"
METADATA_PATH = os.path.join(BASE_DIR, "processed/processed_metadata.csv")
MEL_PATH = os.path.join(BASE_DIR, "processed/mels_128x1024")
EMB_DIR = os.path.join(BASE_DIR, "embeddings_labse_v2")
AUDIO_STATS_PATH = os.path.join(BASE_DIR, "processed/audio_stats_data_stats_data.csv")
OUT_DIR = os.path.join(BASE_DIR, "results_hard")

os.makedirs(OUT_DIR, exist_ok=True)

# Ultra-light configuration
CONFIG = {
    'batch_size': 4,
    'num_workers': 0,
    'epochs': 20,
    'learning_rate': 1e-4,
    'latent_dim': 32,
    'beta': 4.0,
    'n_clusters': 8,
    'patience': 5,
    'mel_time_dim': 128,
    'max_train_samples': 1500,
    'max_eval_samples': 1000
}


def clear_memory():
    """Aggressively clear memory to prevent GPU/CPU memory issues."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_tsne(n_components=2, perplexity=30, random_state=42, max_iter=300):
    """Create TSNE object compatible with different sklearn versions."""
    try:
        # Try newer sklearn parameter name
        return TSNE(n_components=n_components, perplexity=perplexity,
                   random_state=random_state, max_iter=max_iter)
    except TypeError:
        # Fall back to older sklearn parameter name
        return TSNE(n_components=n_components, perplexity=perplexity,
                   random_state=random_state, n_iter=max_iter)


# ============================================================================
# DATASET CLASS
# ============================================================================
class MusicDataset(Dataset):
    """Dataset class for loading music data (mel spectrograms, lyrics embeddings, tabular features)."""

    def __init__(self, track_ids, mel_path, lyrics_path, tabular_features):
        self.track_ids = list(track_ids)
        self.mel_path = mel_path
        self.lyrics_path = lyrics_path
        self.tabular_features = tabular_features

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        track_id = self.track_ids[idx]

        # Load mel spectrogram - heavily downsampled
        mel_file = os.path.join(self.mel_path, f"{track_id}.npz")
        mel_data = np.load(mel_file)['mel']

        # Downsample to (64, 128)
        mel_data = mel_data[::2, :]  # 64 bins

        if mel_data.shape[1] >= CONFIG['mel_time_dim']:
            step = max(1, mel_data.shape[1] // CONFIG['mel_time_dim'])
            mel_data = mel_data[:, ::step][:, :CONFIG['mel_time_dim']]
        else:
            pad_width = CONFIG['mel_time_dim'] - mel_data.shape[1]
            mel_data = np.pad(mel_data, ((0, 0), (0, pad_width)), mode='constant')

        # Ensure correct shape
        if mel_data.shape[0] != 64:
            mel_data = mel_data[:64, :]
        if mel_data.shape[1] != CONFIG['mel_time_dim']:
            mel_data = mel_data[:, :CONFIG['mel_time_dim']]

        mel_tensor = torch.FloatTensor(mel_data).unsqueeze(0)

        # Load lyrics embedding - reduce dimension
        lyrics_file = os.path.join(self.lyrics_path, f"{track_id}.npy")
        lyrics_data = np.load(lyrics_file)
        if len(lyrics_data.shape) > 1:
            lyrics_data = lyrics_data.flatten()
        lyrics_data = lyrics_data[:256] if len(lyrics_data) > 256 else np.pad(lyrics_data, (0, max(0, 256-len(lyrics_data))))
        lyrics_tensor = torch.FloatTensor(lyrics_data)

        # Tabular features
        tabular_data = self.tabular_features.loc[track_id].values.astype(np.float32)
        tabular_tensor = torch.FloatTensor(tabular_data)

        return mel_tensor, tabular_tensor, lyrics_tensor, track_id


def create_dataloader(track_ids, mel_path, lyrics_path, tabular_features,
                      batch_size=4, shuffle=True):
    """Create DataLoader for music dataset."""
    dataset = MusicDataset(track_ids, mel_path, lyrics_path, tabular_features)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                           num_workers=0, pin_memory=False)
    return dataloader


# ============================================================================
# ULTRA-LIGHT MODEL DEFINITIONS
# ============================================================================

class LightBetaVAE(nn.Module):
    """Ultra-lightweight Beta-VAE for multimodal music data."""

    def __init__(self, tabular_dim=55, lyrics_dim=256, latent_dim=32, beta=4.0):
        super(LightBetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.mel_flat_dim = 512
        self.tabular_dim = tabular_dim
        self.lyrics_dim = lyrics_dim

        # Simple mel encoder - input: (1, 64, 128)
        self.mel_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 4))
        )

        # Simple tabular encoder
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        # Simple lyrics encoder
        self.lyrics_encoder = nn.Sequential(
            nn.Linear(lyrics_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Combined: 512 + 16 + 32 = 560
        combined_dim = self.mel_flat_dim + 16 + 32

        # To latent
        self.fc_hidden = nn.Linear(combined_dim, 64)
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, combined_dim),
            nn.ReLU()
        )

        # Mel decoder
        self.mel_decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )

        # Output projections
        self.tabular_decoder = nn.Linear(16, tabular_dim)
        self.lyrics_decoder = nn.Linear(32, lyrics_dim)

    def encode(self, mel, tabular, lyrics):
        mel_enc = self.mel_encoder(mel).view(mel.size(0), -1)
        tab_enc = self.tabular_encoder(tabular)
        lyr_enc = self.lyrics_encoder(lyrics)

        combined = torch.cat([mel_enc, tab_enc, lyr_enc], dim=1)
        hidden = F.relu(self.fc_hidden(combined))

        return self.fc_mu(hidden), self.fc_logvar(hidden)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_fc(z)

        mel_flat = x[:, :self.mel_flat_dim]
        tab_flat = x[:, self.mel_flat_dim:self.mel_flat_dim+16]
        lyr_flat = x[:, self.mel_flat_dim+16:]

        mel_recon = mel_flat.view(-1, 64, 2, 4)
        mel_recon = self.mel_decoder(mel_recon)
        mel_recon = F.interpolate(mel_recon, size=(64, 128), mode='bilinear', align_corners=False)

        tab_recon = self.tabular_decoder(tab_flat)
        lyr_recon = self.lyrics_decoder(lyr_flat)

        return mel_recon, tab_recon, lyr_recon

    def forward(self, mel, tabular, lyrics):
        mu, logvar = self.encode(mel, tabular, lyrics)
        z = self.reparameterize(mu, logvar)
        mel_recon, tab_recon, lyr_recon = self.decode(z)
        return mel_recon, tab_recon, lyr_recon, mu, logvar, z

    def loss_function(self, mel_recon, tab_recon, lyr_recon, mel, tabular, lyrics, mu, logvar):
        recon = F.mse_loss(mel_recon, mel) + 0.5 * F.mse_loss(tab_recon, tabular) + 0.5 * F.mse_loss(lyr_recon, lyrics)
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + self.beta * kld, recon, kld


class LightCVAE(nn.Module):
    """Ultra-lightweight CVAE for multimodal music data."""

    def __init__(self, tabular_dim=55, lyrics_dim=256, latent_dim=32):
        super(LightCVAE, self).__init__()

        self.latent_dim = latent_dim
        self.mel_flat_dim = 512
        self.tabular_dim = tabular_dim
        self.lyrics_dim = lyrics_dim

        # Mel encoder
        self.mel_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 4))
        )

        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.lyrics_encoder = nn.Sequential(
            nn.Linear(lyrics_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        combined_dim = self.mel_flat_dim + 16 + 32

        self.fc_mu = nn.Linear(combined_dim, latent_dim)
        self.fc_logvar = nn.Linear(combined_dim, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, combined_dim)

        self.mel_decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )

        self.tabular_decoder = nn.Linear(16, tabular_dim)
        self.lyrics_decoder = nn.Linear(32, lyrics_dim)

    def encode(self, mel, tabular, lyrics):
        mel_enc = self.mel_encoder(mel).view(mel.size(0), -1)
        tab_enc = self.tabular_encoder(tabular)
        lyr_enc = self.lyrics_encoder(lyrics)
        combined = torch.cat([mel_enc, tab_enc, lyr_enc], dim=1)
        return self.fc_mu(combined), self.fc_logvar(combined)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        x = F.relu(self.decoder_fc(z))

        mel_recon = x[:, :self.mel_flat_dim].view(-1, 64, 2, 4)
        mel_recon = self.mel_decoder(mel_recon)
        mel_recon = F.interpolate(mel_recon, size=(64, 128), mode='bilinear', align_corners=False)

        tab_recon = self.tabular_decoder(x[:, self.mel_flat_dim:self.mel_flat_dim+16])
        lyr_recon = self.lyrics_decoder(x[:, self.mel_flat_dim+16:])

        return mel_recon, tab_recon, lyr_recon

    def forward(self, mel, tabular, lyrics):
        mu, logvar = self.encode(mel, tabular, lyrics)
        z = self.reparameterize(mu, logvar)
        mel_recon, tab_recon, lyr_recon = self.decode(z)
        return mel_recon, tab_recon, lyr_recon, mu, logvar, z

    def loss_function(self, mel_recon, tab_recon, lyr_recon, mel, tabular, lyrics, mu, logvar):
        recon = F.mse_loss(mel_recon, mel) + 0.5 * F.mse_loss(tab_recon, tabular) + 0.5 * F.mse_loss(lyr_recon, lyrics)
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + kld, recon, kld


class LightAutoencoder(nn.Module):
    """Ultra-lightweight Autoencoder for multimodal music data."""

    def __init__(self, tabular_dim=55, lyrics_dim=256, latent_dim=32):
        super(LightAutoencoder, self).__init__()

        self.latent_dim = latent_dim
        self.mel_flat_dim = 512
        self.tabular_dim = tabular_dim
        self.lyrics_dim = lyrics_dim

        self.mel_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 4))
        )

        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.lyrics_encoder = nn.Sequential(
            nn.Linear(lyrics_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        combined_dim = self.mel_flat_dim + 16 + 32

        self.encoder = nn.Linear(combined_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, combined_dim)

        self.mel_decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )

        self.tabular_decoder = nn.Linear(16, tabular_dim)
        self.lyrics_decoder = nn.Linear(32, lyrics_dim)

    def encode(self, mel, tabular, lyrics):
        mel_enc = self.mel_encoder(mel).view(mel.size(0), -1)
        tab_enc = self.tabular_encoder(tabular)
        lyr_enc = self.lyrics_encoder(lyrics)
        combined = torch.cat([mel_enc, tab_enc, lyr_enc], dim=1)
        return self.encoder(combined)

    def decode(self, z):
        x = F.relu(self.decoder(z))

        mel_recon = x[:, :self.mel_flat_dim].view(-1, 64, 2, 4)
        mel_recon = self.mel_decoder(mel_recon)
        mel_recon = F.interpolate(mel_recon, size=(64, 128), mode='bilinear', align_corners=False)

        tab_recon = self.tabular_decoder(x[:, self.mel_flat_dim:self.mel_flat_dim+16])
        lyr_recon = self.lyrics_decoder(x[:, self.mel_flat_dim+16:])

        return mel_recon, tab_recon, lyr_recon

    def forward(self, mel, tabular, lyrics):
        latent = self.encode(mel, tabular, lyrics)
        mel_recon, tab_recon, lyr_recon = self.decode(latent)
        return mel_recon, tab_recon, lyr_recon, latent

    def loss_function(self, mel_recon, tab_recon, lyr_recon, mel, tabular, lyrics):
        return F.mse_loss(mel_recon, mel) + 0.5 * F.mse_loss(tab_recon, tabular) + 0.5 * F.mse_loss(lyr_recon, lyrics)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_model(model, train_loader, val_loader, epochs, lr, device, model_type='vae', patience=5):
    """Generic training function with memory management and early stopping."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        batch_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_data in pbar:
            try:
                mel, tabular, lyrics, _ = batch_data
                mel = mel.to(device)
                tabular = tabular.to(device)
                lyrics = lyrics.to(device)

                optimizer.zero_grad()

                if model_type == 'vae':
                    mel_r, tab_r, lyr_r, mu, logvar, z = model(mel, tabular, lyrics)
                    loss, _, _ = model.loss_function(mel_r, tab_r, lyr_r, mel, tabular, lyrics, mu, logvar)
                else:
                    mel_r, tab_r, lyr_r, latent = model(mel, tabular, lyrics)
                    loss = model.loss_function(mel_r, tab_r, lyr_r, mel, tabular, lyrics)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                batch_count += 1
                pbar.set_postfix({'loss': f'{loss.item():.2f}'})

            except Exception as e:
                print(f"Error in batch: {e}")
                continue

        if batch_count > 0:
            train_loss /= batch_count
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        val_count = 0
        with torch.no_grad():
            for batch_data in val_loader:
                try:
                    mel, tabular, lyrics, _ = batch_data
                    mel = mel.to(device)
                    tabular = tabular.to(device)
                    lyrics = lyrics.to(device)

                    if model_type == 'vae':
                        mel_r, tab_r, lyr_r, mu, logvar, z = model(mel, tabular, lyrics)
                        loss, _, _ = model.loss_function(mel_r, tab_r, lyr_r, mel, tabular, lyrics, mu, logvar)
                    else:
                        mel_r, tab_r, lyr_r, latent = model(mel, tabular, lyrics)
                        loss = model.loss_function(mel_r, tab_r, lyr_r, mel, tabular, lyrics)

                    val_loss += loss.item()
                    val_count += 1
                except Exception as e:
                    continue

        if val_count > 0:
            val_loss /= val_count
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}: Train={train_loss:.2f}, Val={val_loss:.2f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                if best_state:
                    model.load_state_dict(best_state)
                break

        clear_memory()

    return train_losses, val_losses


# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================

def extract_latents(model, dataloader, device, model_type='vae', max_samples=1000):
    """Extract latent representations with sample limit."""
    model.eval()
    latents = []
    track_ids_list = []
    count = 0

    with torch.no_grad():
        for batch_data in dataloader:
            if count >= max_samples:
                break

            try:
                mel, tabular, lyrics, track_ids = batch_data
                mel = mel.to(device)
                tabular = tabular.to(device)
                lyrics = lyrics.to(device)

                if model_type == 'vae':
                    mu, _ = model.encode(mel, tabular, lyrics)
                    latents.append(mu.cpu().numpy())
                else:
                    latent = model.encode(mel, tabular, lyrics)
                    latents.append(latent.cpu().numpy())

                track_ids_list.extend(list(track_ids))
                count += mel.size(0)
            except Exception as e:
                print(f"Error extracting: {e}")
                continue

    if len(latents) == 0:
        return np.array([]), []

    return np.vstack(latents), track_ids_list


def extract_features_for_pca(dataloader, max_samples=800):
    """Extract flattened features for PCA."""
    features = []
    count = 0

    for batch_data in dataloader:
        if count >= max_samples:
            break

        try:
            mel, tabular, lyrics, _ = batch_data
            batch_size = mel.size(0)

            # Flatten and subsample
            mel_flat = mel.view(batch_size, -1).numpy()[:, ::20]
            tab_np = tabular.numpy()
            lyr_np = lyrics.numpy()

            combined = np.concatenate([mel_flat, tab_np, lyr_np], axis=1)
            features.append(combined)
            count += batch_size
        except Exception as e:
            print(f"Error in PCA extraction: {e}")
            continue

    if len(features) == 0:
        return np.array([])

    return np.vstack(features)


# ============================================================================
# METRICS AND VISUALIZATION
# ============================================================================

def compute_metrics(latents, n_clusters=8):
    """Compute clustering metrics."""
    if len(latents) < n_clusters:
        print(f"Warning: Not enough samples ({len(latents)}) for {n_clusters} clusters")
        n_clusters = max(2, len(latents) // 2)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(latents)

    return {
        'silhouette_score': silhouette_score(latents, labels),
        'calinski_harabasz_score': calinski_harabasz_score(latents, labels),
        'davies_bouldin_score': davies_bouldin_score(latents, labels),
        'inertia': kmeans.inertia_
    }, labels


def compute_disentanglement(model, dataloader, device, latent_dim):
    """Compute disentanglement metrics for VAE."""
    model.eval()
    all_mu, all_logvar = [], []

    with torch.no_grad():
        for batch_data in dataloader:
            try:
                mel, tabular, lyrics, _ = batch_data
                mel = mel.to(device)
                tabular = tabular.to(device)
                lyrics = lyrics.to(device)

                mu, logvar = model.encode(mel, tabular, lyrics)
                all_mu.append(mu.cpu().numpy())
                all_logvar.append(logvar.cpu().numpy())
            except:
                continue

    if len(all_mu) == 0:
        return {
            'avg_kl_per_dim': np.zeros(latent_dim),
            'var_per_dim': np.zeros(latent_dim),
            'active_units': 0,
            'total_dims': latent_dim
        }

    all_mu = np.vstack(all_mu)
    all_logvar = np.vstack(all_logvar)

    kl_per_dim = 0.5 * (all_mu**2 + np.exp(all_logvar) - all_logvar - 1)
    var_per_dim = np.var(all_mu, axis=0)

    return {
        'avg_kl_per_dim': np.mean(kl_per_dim, axis=0),
        'var_per_dim': var_per_dim,
        'active_units': int(np.sum(var_per_dim > 0.01)),
        'total_dims': latent_dim
    }


def plot_latent_space(latents, labels, title, save_path, max_samples=1000):
    """Plot t-SNE of latent space."""
    if len(latents) == 0:
        print(f"Warning: No latents to plot for {title}")
        return

    plt.figure(figsize=(10, 8))

    n = min(len(latents), max_samples)
    idx = np.random.choice(len(latents), n, replace=False)
    latents_s = latents[idx]
    labels_s = labels[idx] if labels is not None else None

    perplexity = min(30, n - 1)
    if perplexity < 5:
        perplexity = 5

    tsne = get_tsne(n_components=2, perplexity=perplexity, random_state=42, max_iter=300)

    try:
        latents_2d = tsne.fit_transform(latents_s)

        if labels_s is not None:
            scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels_s, cmap='tab10', alpha=0.6, s=15)
            plt.colorbar(scatter, label='Cluster')
        else:
            plt.scatter(latents_2d[:, 0], latents_2d[:, 1], alpha=0.6, s=15)

        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"Saved: {save_path}")
    except Exception as e:
        print(f"Error plotting {title}: {e}")
        plt.close()


def plot_clusters(latents, labels, title, save_path, max_samples=1000):
    """Plot clusters."""
    if len(latents) == 0:
        print(f"Warning: No latents to plot for {title}")
        return

    plt.figure(figsize=(12, 8))

    n = min(len(latents), max_samples)
    idx = np.random.choice(len(latents), n, replace=False)
    latents_s = latents[idx]
    labels_s = labels[idx]

    perplexity = min(30, n - 1)
    if perplexity < 5:
        perplexity = 5

    tsne = get_tsne(n_components=2, perplexity=perplexity, random_state=42, max_iter=300)

    try:
        latents_2d = tsne.fit_transform(latents_s)

        for label in np.unique(labels_s):
            mask = labels_s == label
            plt.scatter(latents_2d[mask, 0], latents_2d[mask, 1], label=f'C{label}', alpha=0.6, s=20)

        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"Saved: {save_path}")
    except Exception as e:
        print(f"Error plotting {title}: {e}")
        plt.close()


def plot_training_curves(histories, save_path):
    """Plot training comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    colors = {'Beta-VAE': 'blue', 'CVAE': 'green', 'Autoencoder': 'red'}

    for name, (train_loss, val_loss) in histories.items():
        if len(train_loss) > 0:
            epochs = range(1, len(train_loss) + 1)
            axes[0].plot(epochs, train_loss, label=name, color=colors.get(name, 'black'))
            axes[1].plot(epochs, val_loss, label=name, color=colors.get(name, 'black'))

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Train Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Val Loss')
    axes[1].set_title('Validation Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Saved: {save_path}")


def plot_metrics_comparison(metrics_dict, save_path):
    """Plot metrics comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    models = list(metrics_dict.keys())
    colors = ['blue', 'green', 'red', 'purple'][:len(models)]

    metrics_names = ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score', 'inertia']
    titles = ['Silhouette (Higher=Better)', 'Calinski-Harabasz (Higher=Better)',
              'Davies-Bouldin (Lower=Better)', 'Inertia (Lower=Better)']

    for ax, metric, title in zip(axes.flat, metrics_names, titles):
        values = [metrics_dict[m][metric] for m in models]
        ax.bar(models, values, color=colors)
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Saved: {save_path}")


def plot_disentanglement(beta_vae_d, cvae_d, save_path):
    """Plot disentanglement analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].bar(range(len(beta_vae_d['avg_kl_per_dim'])), beta_vae_d['avg_kl_per_dim'], color='blue', alpha=0.7)
    axes[0, 0].set_title('Beta-VAE: KL per Dimension')
    axes[0, 0].set_xlabel('Dimension')

    axes[0, 1].bar(range(len(cvae_d['avg_kl_per_dim'])), cvae_d['avg_kl_per_dim'], color='green', alpha=0.7)
    axes[0, 1].set_title('CVAE: KL per Dimension')
    axes[0, 1].set_xlabel('Dimension')

    axes[1, 0].bar(range(len(beta_vae_d['var_per_dim'])), beta_vae_d['var_per_dim'], color='blue', alpha=0.7)
    axes[1, 0].axhline(0.01, color='r', linestyle='--')
    axes[1, 0].set_title(f'Beta-VAE: Variance (Active: {beta_vae_d["active_units"]}/{beta_vae_d["total_dims"]})')

    axes[1, 1].bar(range(len(cvae_d['var_per_dim'])), cvae_d['var_per_dim'], color='green', alpha=0.7)
    axes[1, 1].axhline(0.01, color='r', linestyle='--')
    axes[1, 1].set_title(f'CVAE: Variance (Active: {cvae_d["active_units"]}/{cvae_d["total_dims"]})')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Saved: {save_path}")


def plot_reconstruction(model, dataloader, device, save_path, model_type='vae'):
    """Plot reconstruction examples."""
    model.eval()

    try:
        batch_data = next(iter(dataloader))
        mel, tabular, lyrics, track_ids = batch_data
        mel = mel.to(device)
        tabular = tabular.to(device)
        lyrics = lyrics.to(device)

        with torch.no_grad():
            if model_type == 'vae':
                mel_r, _, _, _, _, _ = model(mel, tabular, lyrics)
            else:
                mel_r, _, _, _ = model(mel, tabular, lyrics)

        n = min(3, mel.size(0))
        fig, axes = plt.subplots(n, 2, figsize=(10, 3*n))

        if n == 1:
            axes = axes.reshape(1, -1)

        for i in range(n):
            axes[i, 0].imshow(mel[i, 0].cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
            axes[i, 0].set_title(f'Original - {track_ids[i]}')
            axes[i, 1].imshow(mel_r[i, 0].cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
            axes[i, 1].set_title('Reconstructed')

        plt.suptitle(f'{model_type.upper()} Reconstruction')
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"Saved: {save_path}")
    except Exception as e:
        print(f"Error plotting reconstruction: {e}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function to run the complete multimodal music analysis pipeline."""
    print("=" * 70)
    print("MULTIMODAL MUSIC ANALYSIS - ULTRA LIGHT VERSION")
    print("=" * 70)

    # Load data
    print("\n[1/8] Loading data...")
    try:
        metadata = pd.read_csv(METADATA_PATH)
        audio_stats = pd.read_csv(AUDIO_STATS_PATH)
        data = pd.merge(metadata, audio_stats, on='TrackID', how='inner')
        print(f"Data shape: {data.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Encode features
    if 'RawLanguage' in data.columns:
        le = LabelEncoder()
        data['RawLanguage_encoded'] = le.fit_transform(data['RawLanguage'].fillna('Unknown'))
    else:
        data['RawLanguage_encoded'] = 0

    if 'PrimaryGenre' in data.columns:
        le = LabelEncoder()
        data['PrimaryGenre_encoded'] = le.fit_transform(data['PrimaryGenre'].fillna('Unknown'))
    else:
        data['PrimaryGenre_encoded'] = 0

    for col in ['Duration_sec', 'sample_rate', 'sentiment']:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median())
        else:
            data[col] = 0

    # Prepare tabular features
    audio_cols = [c for c in audio_stats.columns if c != 'TrackID']
    tab_cols = audio_cols + ['RawLanguage_encoded', 'PrimaryGenre_encoded', 'Duration_sec', 'sample_rate', 'sentiment']
    tab_cols = [c for c in tab_cols if c in data.columns]

    tabular_df = data[['TrackID'] + tab_cols].set_index('TrackID').fillna(0)
    scaler = StandardScaler()
    tabular_scaled = pd.DataFrame(
        scaler.fit_transform(tabular_df),
        index=tabular_df.index,
        columns=tabular_df.columns
    )

    TABULAR_DIM = len(tab_cols)
    print(f"Tabular dim: {TABULAR_DIM}")

    # Find valid tracks
    track_ids = []
    for tid in tqdm(tabular_scaled.index, desc="Checking tracks"):
        mel_path = os.path.join(MEL_PATH, f"{tid}.npz")
        emb_path = os.path.join(EMB_DIR, f"{tid}.npy")
        if os.path.exists(mel_path) and os.path.exists(emb_path):
            track_ids.append(tid)

    print(f"Valid tracks: {len(track_ids)}")

    if len(track_ids) == 0:
        print("ERROR: No valid tracks found!")
        return

    # Limit samples
    if len(track_ids) > CONFIG['max_train_samples']:
        track_ids = track_ids[:CONFIG['max_train_samples']]
        print(f"Limited to {len(track_ids)} samples")

    tabular_scaled = tabular_scaled.loc[track_ids]

    # Split
    train_ids, val_ids = train_test_split(track_ids, test_size=0.2, random_state=42)
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

    # Create loaders
    train_loader = create_dataloader(train_ids, MEL_PATH, EMB_DIR, tabular_scaled,
                                     batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = create_dataloader(val_ids, MEL_PATH, EMB_DIR, tabular_scaled,
                                   batch_size=CONFIG['batch_size'], shuffle=False)
    eval_loader = create_dataloader(track_ids, MEL_PATH, EMB_DIR, tabular_scaled,
                                    batch_size=CONFIG['batch_size'], shuffle=False)

    histories = {}
    all_metrics = {}
    all_labels = {}

    # ========== TRAIN BETA-VAE ==========
    print("\n[2/8] Training Beta-VAE...")
    beta_vae = LightBetaVAE(tabular_dim=TABULAR_DIM, latent_dim=CONFIG['latent_dim'], beta=CONFIG['beta'])
    print(f"Parameters: {sum(p.numel() for p in beta_vae.parameters()):,}")

    histories['Beta-VAE'] = train_model(
        beta_vae, train_loader, val_loader,
        CONFIG['epochs'], CONFIG['learning_rate'], device, 'vae', CONFIG['patience']
    )

    # Extract and evaluate
    beta_latents, _ = extract_latents(beta_vae, eval_loader, device, 'vae', CONFIG['max_eval_samples'])

    if len(beta_latents) > 0:
        all_metrics['Beta-VAE'], all_labels['Beta-VAE'] = compute_metrics(beta_latents, CONFIG['n_clusters'])
        beta_disentangle = compute_disentanglement(beta_vae, eval_loader, device, CONFIG['latent_dim'])

        # Save model and plots
        torch.save(beta_vae.state_dict(), os.path.join(OUT_DIR, 'beta_vae_model.pth'))
        plot_latent_space(beta_latents, all_labels['Beta-VAE'], 'Beta-VAE Latent Space',
                         os.path.join(OUT_DIR, 'beta_vae_latent.png'))
        plot_clusters(beta_latents, all_labels['Beta-VAE'], 'Beta-VAE Clusters',
                     os.path.join(OUT_DIR, 'beta_vae_clusters.png'))
        plot_reconstruction(beta_vae, val_loader, device, os.path.join(OUT_DIR, 'reconstruction_examples.png'), 'vae')
    else:
        print("Warning: No latents extracted for Beta-VAE")
        beta_disentangle = {'avg_kl_per_dim': np.zeros(CONFIG['latent_dim']),
                           'var_per_dim': np.zeros(CONFIG['latent_dim']),
                           'active_units': 0, 'total_dims': CONFIG['latent_dim']}

    # Clear
    del beta_vae
    if 'beta_latents' in dir():
        del beta_latents
    clear_memory()

    # ========== TRAIN CVAE ==========
    print("\n[3/8] Training CVAE...")
    cvae = LightCVAE(tabular_dim=TABULAR_DIM, latent_dim=CONFIG['latent_dim'])
    print(f"Parameters: {sum(p.numel() for p in cvae.parameters()):,}")

    histories['CVAE'] = train_model(
        cvae, train_loader, val_loader,
        CONFIG['epochs'], CONFIG['learning_rate'], device, 'vae', CONFIG['patience']
    )

    cvae_latents, _ = extract_latents(cvae, eval_loader, device, 'vae', CONFIG['max_eval_samples'])

    if len(cvae_latents) > 0:
        all_metrics['CVAE'], all_labels['CVAE'] = compute_metrics(cvae_latents, CONFIG['n_clusters'])
        cvae_disentangle = compute_disentanglement(cvae, eval_loader, device, CONFIG['latent_dim'])

        torch.save(cvae.state_dict(), os.path.join(OUT_DIR, 'cvae_model.pth'))
        plot_latent_space(cvae_latents, all_labels['CVAE'], 'CVAE Latent Space',
                         os.path.join(OUT_DIR, 'cvae_latent.png'))
        plot_clusters(cvae_latents, all_labels['CVAE'], 'CVAE Clusters',
                     os.path.join(OUT_DIR, 'cvae_clusters.png'))
    else:
        print("Warning: No latents extracted for CVAE")
        cvae_disentangle = {'avg_kl_per_dim': np.zeros(CONFIG['latent_dim']),
                           'var_per_dim': np.zeros(CONFIG['latent_dim']),
                           'active_units': 0, 'total_dims': CONFIG['latent_dim']}

    del cvae
    if 'cvae_latents' in dir():
        del cvae_latents
    clear_memory()

    # ========== TRAIN AUTOENCODER ==========
    print("\n[4/8] Training Autoencoder...")
    ae = LightAutoencoder(tabular_dim=TABULAR_DIM, latent_dim=CONFIG['latent_dim'])
    print(f"Parameters: {sum(p.numel() for p in ae.parameters()):,}")

    histories['Autoencoder'] = train_model(
        ae, train_loader, val_loader,
        CONFIG['epochs'], CONFIG['learning_rate'], device, 'ae', CONFIG['patience']
    )

    ae_latents, _ = extract_latents(ae, eval_loader, device, 'ae', CONFIG['max_eval_samples'])

    if len(ae_latents) > 0:
        all_metrics['Autoencoder'], all_labels['Autoencoder'] = compute_metrics(ae_latents, CONFIG['n_clusters'])
        torch.save(ae.state_dict(), os.path.join(OUT_DIR, 'autoencoder_model.pth'))
    else:
        print("Warning: No latents extracted for Autoencoder")

    del ae
    if 'ae_latents' in dir():
        del ae_latents
    clear_memory()

    # ========== PCA ==========
    print("\n[5/8] Running PCA...")
    pca_features = extract_features_for_pca(eval_loader, max_samples=800)

    if len(pca_features) > 0:
        print(f"PCA features shape: {pca_features.shape}")

        pca_scaler = StandardScaler()
        pca_scaled = pca_scaler.fit_transform(pca_features)

        n_components = min(CONFIG['latent_dim'], pca_scaled.shape[1], pca_scaled.shape[0] - 1)
        pca = PCA(n_components=n_components)
        pca_latents = pca.fit_transform(pca_scaled)
        print(f"PCA latents shape: {pca_latents.shape}")

        all_metrics['PCA'], all_labels['PCA'] = compute_metrics(pca_latents, CONFIG['n_clusters'])

        plot_latent_space(pca_latents, all_labels['PCA'], 'PCA Latent Space',
                         os.path.join(OUT_DIR, 'pca_latent.png'))

        del pca_features, pca_scaled, pca_latents
    else:
        print("Warning: Could not extract PCA features")

    clear_memory()

    # ========== GENERATE PLOTS ==========
    print("\n[6/8] Generating comparison plots...")

    if len(histories) > 0:
        plot_training_curves(histories, os.path.join(OUT_DIR, 'training_comparison.png'))

    if len(all_metrics) > 0:
        plot_metrics_comparison(all_metrics, os.path.join(OUT_DIR, 'metrics_comparison.png'))

    if 'beta_disentangle' in dir() and 'cvae_disentangle' in dir():
        plot_disentanglement(beta_disentangle, cvae_disentangle, os.path.join(OUT_DIR, 'disentanglement_analysis.png'))

    # ========== SAVE METRICS ==========
    print("\n[7/8] Saving metrics...")
    if len(all_metrics) > 0:
        metrics_df = pd.DataFrame(all_metrics).T
        metrics_df.to_csv(os.path.join(OUT_DIR, 'clustering_metrics.csv'))
        print(f"Saved: {os.path.join(OUT_DIR, 'clustering_metrics.csv')}")

    # ========== SUMMARY ==========
    print("\n[8/8] Complete!")
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    if len(all_metrics) > 0:
        print("\nClustering Metrics:")
        metrics_df = pd.DataFrame(all_metrics).T
        print(metrics_df.to_string())

    if 'beta_disentangle' in dir():
        print(f"\nBeta-VAE Active Units: {beta_disentangle['active_units']}/{beta_disentangle['total_dims']}")
    if 'cvae_disentangle' in dir():
        print(f"CVAE Active Units: {cvae_disentangle['active_units']}/{cvae_disentangle['total_dims']}")

    print(f"\nAll results saved to: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()