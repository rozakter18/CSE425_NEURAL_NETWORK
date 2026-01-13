# -*- coding: utf-8 -*-
"""
Medium VAE Training Script

This script trains a VAE on mel spectrograms and lyrics embeddings for medium task.

Usage:
1. Ensure data files are available.
2. Run: python medium_vae.py
3. Trained model and latents will be saved.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Local paths
BASE_DIR = "C:/Users/YamalxMessi/Desktop/Project"
META_PATH = os.path.join(BASE_DIR, "processed/processed_metadata.csv")

MELS_DIR = os.path.join(BASE_DIR, "processed/mels_128x1024")
EMB_DIR = os.path.join(BASE_DIR, "embeddings_labse_v2")

OUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

VALID_LANGS = ["English", "Bangla", "Korean"]

LATENT_DIM = 32
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 50
PATIENCE = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_H, TARGET_W = 128, 1024

def load_meta(path, valid_langs):
    """Load and filter metadata by languages."""
    meta = pd.read_csv(path)
    print("Raw metadata:", meta.shape)

    lang_col = "RawLanguage"
    if lang_col in meta.columns:
        meta = meta[meta[lang_col].isin(valid_langs)].reset_index(drop=True)
        print(f"Metadata after language filter ({valid_langs}):", meta.shape)
    else:
        print(f"[WARN] Column '{lang_col}' not found – skipping language filter.")

    assert "TrackID" in meta.columns, "Meta must contain 'TrackID' column."
    return meta

def load_mels_resized(meta, mel_dir):
    """Load and resize mel spectrograms."""
    ids_audio = []
    bad = []

    for tid in tqdm(meta["TrackID"].astype(str).values, desc="Loading mel-spectrograms"):
        path = os.path.join(mel_dir, f"{tid}.npz")
        if not os.path.exists(path):
            bad.append((tid, "missing"))
            continue
        try:
            d = np.load(path)
            mel = d["mel"].astype(np.float32)
            H, W = mel.shape

            if H < TARGET_H:
                mel = np.pad(mel, ((0, TARGET_H - H), (0, 0)))
            elif H > TARGET_H:
                mel = mel[:TARGET_H, :]

            if W < TARGET_W:
                mel = np.pad(mel, ((0, 0), (0, TARGET_W - W)))
            elif W > TARGET_W:
                mel = mel[:, :TARGET_W]

            mel = mel[None, ...]
            mels.append(mel)
            ids_audio.append(tid)
        except Exception as e:
            bad.append((tid, f"error: {e}"))

    print(f"Loaded mels: {len(mels)}; bad files: {len(bad)}")
    if len(mels) == 0:
        raise RuntimeError("No mel files loaded – check MELS_DIR")

    X_audio = np.stack(mels, axis=0)
    print("Audio array shape:", X_audio.shape)
    return X_audio, ids_audio

def build_labse_dict(emb_dir):

    id_to_emb = {}
    files = sorted([f for f in os.listdir(emb_dir) if f.endswith(".npz")])
    print("Found LABSE chunks:", files)

    for f in files:
        path = os.path.join(emb_dir, f)
        d = np.load(path, allow_pickle=True)
        ids = d["ids"]
        emb = d["emb"]

        for tid, vec in zip(ids, emb):
            tid_str = str(tid)
            id_to_emb[tid_str] = vec.astype(np.float32)

    print("Total unique lyric embeddings:", len(id_to_emb))
    return id_to_emb

def load_embeddings_for_meta(meta, emb_dir):

    emb_list = []
    ids_text = []
    emb_dim = None

    for tid in tqdm(meta["TrackID"].values, desc="Loading LaBSE embeddings"):
        path = os.path.join(emb_dir, f"{tid}.npy")
        if os.path.exists(path):
            emb = np.load(path)
            if emb_dim is None:
                emb_dim = emb.shape[0]
        else:
            if emb_dim is None:
                emb_dim = 768
            emb = np.zeros(emb_dim, dtype=np.float32)

        emb_list.append(emb)
        ids_text.append(tid)

    X_text = np.stack(emb_list, axis=0).astype(np.float32)
    print("Loaded lyrics embeddings:", X_text.shape)
    return X_text, ids_text

def align_modalities(meta, X_audio, ids_audio, X_text, ids_text):

    set_a = set(ids_audio)
    set_t = set(ids_text)
    keep = sorted(list(set_a & set_t))
    print("Intersection size:", len(keep))

    idx_a = {tid: i for i, tid in enumerate(ids_audio)}
    idx_t = {tid: i for i, tid in enumerate(ids_text)}

    rows_a = [idx_a[t] for t in keep]
    rows_t = [idx_t[t] for t in keep]

    X_audio_aligned = X_audio[rows_a]
    X_text_aligned  = X_text[rows_t]
    meta_aligned    = meta[meta["TrackID"].astype(str).isin(keep)].reset_index(drop=True)

    print("Aligned audio :", X_audio_aligned.shape)
    print("Aligned lyrics:", X_text_aligned.shape)
    print("Aligned meta  :", meta_aligned.shape)

    return X_audio_aligned, X_text_aligned, meta_aligned

class HybridAudioTextDataset(Dataset):
    def __init__(self, X_audio, X_text):
        assert len(X_audio) == len(X_text)
        self.Xa = X_audio
        self.Xt = X_text

    def __len__(self):
        return len(self.Xa)

    def __getitem__(self, idx):
        a = self.Xa[idx]
        t = self.Xt[idx]
        return (
            torch.from_numpy(a).float(),
            torch.from_numpy(t).float()
        )

class ConvAudioEncoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.flat_dim = 256 * 8 * 64
        self.fc_mu    = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar= nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x):
        b = x.size(0)
        h = self.conv(x)
        h = h.view(b, -1)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class LyricsEncoder(nn.Module):
    def __init__(self, input_dim=768, latent_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.fc_mu    = nn.Linear(256, latent_dim)
        self.fc_logvar= nn.Linear(256, latent_dim)

    def forward(self, x):
        h = self.mlp(x)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class ConvDecoder(nn.Module):
    def __init__(self, fused_latent_dim=64, flat_dim=256*8*64):
        super().__init__()
        self.fc = nn.Linear(fused_latent_dim, flat_dim)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1,   4, stride=2, padding=1),
        )

    def forward(self, z):
        b = z.size(0)
        h = self.fc(z).view(b, 256, 8, 64)
        x_hat = self.deconv(h)
        return x_hat[:, :, :TARGET_H, :TARGET_W]

class HybridVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.audio_enc  = ConvAudioEncoder(latent_dim)
        self.text_enc   = LyricsEncoder(768, latent_dim)
        self.decoder    = ConvDecoder(fused_latent_dim=2*latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_audio, x_text):
        # encoders
        mu_a, logvar_a = self.audio_enc(x_audio)
        mu_t, logvar_t = self.text_enc(x_text)

        z_a = self.reparameterize(mu_a, logvar_a)
        z_t = self.reparameterize(mu_t, logvar_t)

        z = torch.cat([z_a, z_t], dim=1)

        x_hat = self.decoder(z)


        recon_loss = torch.mean((x_hat - x_audio) ** 2)


        kl_a = -0.5 * torch.mean(1 + logvar_a - mu_a.pow(2) - logvar_a.exp())
        kl_t = -0.5 * torch.mean(1 + logvar_t - mu_t.pow(2) - logvar_t.exp())
        kl = kl_a + kl_t

        loss = recon_loss + kl
        return loss, recon_loss, kl, mu_a, mu_t

    @torch.no_grad()
    def encode_fused_mu(self, x_audio, x_text):
        mu_a, _ = self.audio_enc(x_audio)
        mu_t, _ = self.text_enc(x_text)
        z = torch.cat([mu_a, mu_t], dim=1)
        return z

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = None
        self.should_stop= False

    def step(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

def train_vae(X_audio, X_text,EPOCHS=50,PATIENCE=5):
    dataset = HybridAudioTextDataset(X_audio, X_text)
    n_total = len(dataset)
    n_val   = max(1, int(0.1 * n_total))
    n_train = n_total - n_val

    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Train size: {n_train}, Val size: {n_val}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = HybridVAE(LATENT_DIM).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)

    stopper = EarlyStopper(patience=PATIENCE, min_delta=1e-3)
    history = []

    best_state = None
    best_val   = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_recon = 0.0
        train_kl = 0.0

        for xa, xt in train_loader:
            xa = xa.to(DEVICE)
            xt = xt.to(DEVICE)

            opt.zero_grad()
            loss, recon, kl, _, _ = model(xa, xt)
            loss.backward()
            opt.step()

            train_loss  += loss.item() * xa.size(0)
            train_recon += recon.item() * xa.size(0)
            train_kl    += kl.item() * xa.size(0)

        train_loss  /= n_train
        train_recon /= n_train
        train_kl    /= n_train

        # validation
        model.eval()
        val_loss = 0.0
        val_recon = 0.0
        val_kl = 0.0
        with torch.no_grad():
            for xa, xt in val_loader:
                xa = xa.to(DEVICE)
                xt = xt.to(DEVICE)
                loss, recon, kl, _, _ = model(xa, xt)
                val_loss  += loss.item() * xa.size(0)
                val_recon += recon.item() * xa.size(0)
                val_kl    += kl.item() * xa.size(0)

        val_loss  /= n_val
        val_recon /= n_val
        val_kl    /= n_val

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_recon": train_recon,
            "train_kl": train_kl,
            "val_loss": val_loss,
            "val_recon": val_recon,
            "val_kl": val_kl,
        })

        print(f"Epoch {epoch:03d} | "
              f"train: {train_loss:.3f} (R={train_recon:.3f}, KL={train_kl:.3f}) | "
              f"val: {val_loss:.3f} (R={val_recon:.3f}, KL={val_kl:.3f})")

        # early stopping
        stopper.step(val_loss)
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()

        if stopper.should_stop:
            print(f"Early stopping at epoch {epoch}")
            break


    if best_state is not None:
        model.load_state_dict(best_state)

    with open(os.path.join(OUT_DIR, "train_history_medium.json"), "w") as f:
        json.dump(history, f, indent=2)


    torch.save(model.state_dict(), os.path.join(OUT_DIR, "hybrid_conv_vae_medium.pt"))

    return model

def extract_and_save_latent(model, X_audio, X_text):
    dataset = HybridAudioTextDataset(X_audio, X_text)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model.eval()
    zs = []
    with torch.no_grad():
        for xa, xt in loader:
            xa = xa.to(DEVICE)
            xt = xt.to(DEVICE)
            z = model.encode_fused_mu(xa, xt)
            zs.append(z.cpu().numpy())

    Z = np.concatenate(zs, axis=0)
    print("Latent shape (fused mu):", Z.shape)


    Z = np.nan_to_num(Z, nan=0.0, posinf=1e6, neginf=-1e6)
    print("NaNs in VAE latent:", np.isnan(Z).sum())

    np.save(os.path.join(OUT_DIR, "Z_medium.npy"), Z)
    return Z

def pca_baseline(X_text):

    print("\n PCA baseline on lyrics embeddings")
    scaler = StandardScaler()
    X_std  = scaler.fit_transform(X_text)
    print("Standardized shape:", X_std.shape)

    max_k = min(128, X_std.shape[1])
    pca = PCA(n_components=max_k, random_state=42)
    X_pca_full = pca.fit_transform(X_std)
    cum_var = np.cumsum(pca.explained_variance_ratio_)


    xs = np.arange(1, max_k + 1)
    y1, y2 = cum_var[0], cum_var[max_k - 1]
    line_y = y1 + (y2 - y1) * (xs - 1) / (max_k - 1)
    distances = np.abs(cum_var[:max_k] - line_y)
    k_auto = int(xs[np.argmax(distances)])

    print(f"K (auto elbow) = {k_auto}")
    print(f"Cumulative Var @K={k_auto}: {cum_var[k_auto-1]:.3f}")

    Z_pca = X_pca_full[:, :k_auto]

    info = {
        "k_auto": int(k_auto),
        "cum_var_at_k": float(cum_var[k_auto - 1]),
        "explained_variance_ratio": pca.explained_variance_ratio_[:k_auto].tolist()
    }

    Z_pca = np.nan_to_num(Z_pca, nan=0.0, posinf=1e6, neginf=-1e6)
    print("NaNs in PCA:", np.isnan(Z_pca).sum())

    np.save(os.path.join(OUT_DIR, "Z_pca_medium.npy"), Z_pca)
    with open(os.path.join(OUT_DIR, "pca_medium.pkl"), "wb") as f:
        pickle.dump({"model": pca, "scaler": scaler}, f)
    with open(os.path.join(OUT_DIR, "pca_medium_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    return Z_pca, pca, scaler, info

def main():

  meta = load_meta(META_PATH, VALID_LANGS)

  X_audio, ids_audio = load_mels_resized(meta, MELS_DIR)
  X_audio = (X_audio - X_audio.mean()) / (X_audio.std() + 1e-6)

  X_text,  ids_text  = load_embeddings_for_meta(meta, EMB_DIR)
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  X_text = scaler.fit_transform(X_text)

  X_audio_aligned, X_text_aligned, meta_aligned = align_modalities(
      meta, X_audio, ids_audio, X_text, ids_text
  )



  print(X_audio_aligned.max(), X_audio_aligned.min())




  meta_aligned.to_csv(os.path.join(OUT_DIR, "meta_aligned_medium.csv"), index=False)


  model = train_vae(X_audio_aligned, X_text_aligned, EPOCHS=50,PATIENCE=5)


  Z_vae = extract_and_save_latent(model, X_audio_aligned, X_text_aligned)


  Z_pca, pca_model, pca_scaler, pca_info = pca_baseline(X_text_aligned)

  print("\nDone. Saved:")
  print("  - VAE latent:       ", os.path.join(OUT_DIR, "Z_medium.npy"))
  print("  - PCA latent:       ", os.path.join(OUT_DIR, "Z_pca_medium.npy"))
  print("  - VAE model:        ", os.path.join(OUT_DIR, "hybrid_conv_vae_medium.pt"))
  print("  - Aligned metadata: ", os.path.join(OUT_DIR, "meta_aligned_medium.csv"))
  print("  - PCA info/json     ", os.path.join(OUT_DIR, "pca_medium_info.json"))


if __name__ == "__main__":
    main()

import numpy as np

Z_vae = np.load("Project/results/Z_medium.npy")
print("Loaded latent:", Z_vae.shape)

print("latent mean:", Z_vae.mean(axis=0)[:10])
print("latent std :", Z_vae.std(axis=0)[:10])

print("unique rows:", np.unique(Z_vae, axis=0).shape[0])

Z_pca = np.load("Project/results/Z_pca_medium.npy")

import matplotlib.pyplot as plt

plt.figure(figsize=(5,5))
plt.scatter(Z_pca[:,0], Z_pca[:,1], s=3, alpha=0.5)
plt.title("PCA latent scatter")
plt.show()