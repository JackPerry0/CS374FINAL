import os
import numpy as np
import pandas as pd
import torch
import h5py

from torchvision import models, transforms
from tqdm import tqdm
from PIL import Image
from pathlib import Path

ROOT = Path(__file__).resolve().parent

MERGED_FILES = [
    str(ROOT / "merged_optical_tabular_0p1.pkl"),
    str(ROOT / "merged_optical_tabular_0p2.pkl"),
]

H5_IMAGE_PATH = os.environ.get("H5_IMAGE_PATH", str(ROOT / "Galaxy10_DECals_sample.h5"))

TABULAR_FEATURES = [
    "z_best","mag_g","mag_r","mag_z","mag_w1","mag_w2",
    "magerr_g","magerr_r","magerr_z","magerr_w1","magerr_w2",
    "flag_qual","flag_mass","Legacy_Coverage"
]

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print("Using device:", DEVICE)


#TABULAR AUTOENCODER

class TabularAutoencoder(torch.nn.Module):
    def __init__(self, input_dim, emb_dim=32):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, emb_dim)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

def train_tabular_ae(df, emb_dim=32, epochs=50, lr=1e-3):
    X = df[TABULAR_FEATURES].to_numpy(dtype=np.float32)
    # Simple impute: replace nan/inf with column means
    X[~np.isfinite(X)] = np.nan
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    nan_mask = ~np.isfinite(X)
    if nan_mask.any():
        X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    model = TabularAutoencoder(input_dim=X.shape[1], emb_dim=emb_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        x_hat, _ = model(X_t)
        loss = loss_fn(x_hat, X_t)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        _, Z = model(X_t)
    return Z.cpu().numpy()



# H5 IMAGE LOADER

def load_images_from_h5(indices):
    if not Path(H5_IMAGE_PATH).exists():
        raise FileNotFoundError(
            f"H5 file not found: {H5_IMAGE_PATH}\n"
            f"For the sample dataset, run: python make_midterm_sample_data.py\n"
            f"Or set H5_IMAGE_PATH=/path/to/your/full Galaxy10 h5."
        )
    with h5py.File(H5_IMAGE_PATH, "r") as f:
        imgs = f["images"]
        stacked = np.stack([imgs[int(i)] for i in indices], axis=0)
    return stacked



# RESNET IMAGE EMBEDDINGS


class ResNetEmbedder:
    def __init__(self):
        # NOTE: this will download weights the first time if they aren't cached.
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        modules = list(base.children())[:-1]
        self.model = torch.nn.Sequential(*modules).to(DEVICE).eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def embed(self, imgs):
        feats = []
        with torch.no_grad():
            for img in tqdm(imgs, desc="ResNet"):
                x = self.transform(img).unsqueeze(0).to(DEVICE)
                f = self.model(x).squeeze().cpu().numpy()
                feats.append(f)
        return np.stack(feats, axis=0)



# MAIN PROCESSING

def process_file(pkl_path):
    print(f"\n==============================")
    print(f"Processing {pkl_path}")
    print(f"==============================")

    df = pd.read_pickle(pkl_path)
    N = len(df)
    print(f"Loaded {N} rows")

    if not all(f in df.columns for f in TABULAR_FEATURES):
        missing = [f for f in TABULAR_FEATURES if f not in df.columns]
        raise ValueError(f"Missing tabular features: {missing}")

    print("Training autoencoder for tabular features...")
    tab_emb = train_tabular_ae(df)

    print("Extracting matched row indices for images...")
    matched_indices = df.index.values

    print("Loading images from H5...")
    imgs = load_images_from_h5(matched_indices)

    print("Running ResNet embeddings...")
    resnet = ResNetEmbedder()
    img_emb = resnet.embed(imgs)

    np.save(pkl_path.replace(".pkl", "_tabAE.npy"), tab_emb)
    np.save(pkl_path.replace(".pkl", "_imgResNet.npy"), img_emb)

    print("Saved:")
    print("  ", pkl_path.replace(".pkl", "_tabAE.npy"))
    print("  ", pkl_path.replace(".pkl", "_imgResNet.npy"))


def main():
    for pkl_path in MERGED_FILES:
        if not Path(pkl_path).exists():
            print("[warn] missing:", pkl_path)
            continue
        process_file(pkl_path)

if __name__ == "__main__":
    main()
