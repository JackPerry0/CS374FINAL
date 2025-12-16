import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


#CONFIG

PKL_PATH = "merged_optical_tabular_0p1.pkl"

# FULL DATASET (12 features only)
KEY_FEATURES_FULL = [
    "mag_g","mag_r","mag_z","mag_w1","mag_w2",
    "magerr_g","magerr_r","magerr_z","magerr_w1","magerr_w2",
    "optRA","optDec"
]

# ENRICHED DATASET (33 features)
KEY_FEATURES_ENRICHED = [
    "z_best","zphot","zphot_err","flag_qual",
    "mag_g","mag_r","mag_z","mag_w1","mag_w2",
    "magerr_g","magerr_r","magerr_z","magerr_w1","magerr_w2",
    "g_rest","r_rest","z_rest","U_rest","V_rest","J_rest","K_rest",
    "w1_rest","w2_rest",
    "Mass_median","Mass_l68","Mass_u68",
    "r_50","r_50_err",
    "pstar",
    "optRA","optDec"
]

LATENT_DIM = 32
EPOCHS = 50
BATCH = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#AUTOENCODER

class TabularAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z


def train_autoencoder(X):
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    loader = DataLoader(X_t, batch_size=BATCH, shuffle=True)

    model = TabularAE(X.shape[1], LATENT_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for ep in range(EPOCHS):
        for batch in loader:
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"Epoch {ep+1}/{EPOCHS}  Loss={loss.item():.6f}")

    with torch.no_grad():
        _, Z = model(X_t)
        Z = Z.cpu().numpy()

    return Z

#MAIN

def main():
    df = pd.read_pickle(PKL_PATH)

    # FULL DATASET
    
    df_full = df[KEY_FEATURES_FULL].dropna().copy()
    print(f"\nFULL dataset size (12 features): {len(df_full)}")

    X_full = df_full.values.astype(np.float32)
    print("Training FULL autoencoder...")
    full_emb = train_autoencoder(X_full)

    np.save(PKL_PATH.replace(".pkl", "_tabAE_full.npy"), full_emb)
    print("Saved FULL embeddings.")

    # ENRICHED DATASET

    df_enr = df[KEY_FEATURES_ENRICHED].dropna().copy()
    print(f"\nENRICHED dataset size (33 features): {len(df_enr)}")

    X_enr = df_enr.values.astype(np.float32)
    print("Training ENRICHED autoencoder...")
    enr_emb = train_autoencoder(X_enr)

    np.save(PKL_PATH.replace(".pkl", "_tabAE_enriched.npy"), enr_emb)
    print("Saved ENRICHED embeddings.")

    print("\nDONE.")


if __name__ == "__main__":
    main()
