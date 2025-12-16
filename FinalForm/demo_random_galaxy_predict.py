import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, d_in: int, hidden_dims=(256, 128), dropout: float = 0.0):
        super().__init__()
        layers = []
        prev = d_in
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout and dropout > 0:
                layers += [nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_embeddings(emb_path: Path, idx_path: Path):
    Z = np.load(emb_path)
    idx = pd.read_csv(idx_path)
    if "master_index" not in idx.columns:
        raise ValueError(f"{idx_path} missing master_index")
    if len(idx) != Z.shape[0]:
        k = min(len(idx), Z.shape[0])
        Z = Z[:k]
        idx = idx.iloc[:k].reset_index(drop=True)
    mids = idx["master_index"].astype(int).to_numpy()
    return Z.astype(np.float32, copy=False), mids


def invert_to_raw(yhat_t: float, transform: str, y_offset: float) -> float:
    if transform == "none":
        return float(yhat_t)
    if transform == "log1p":
        return float(np.expm1(yhat_t) - y_offset)
    if transform == "log10p1":
        return float((10.0 ** yhat_t - 1.0) - y_offset)
    raise ValueError(f"Unknown transform: {transform}")


def pick_flux_column(df: pd.DataFrame) -> str:
    # prefer common names first
    candidates = ["Total_flux", "total_flux", "int_flux", "Int_flux", "lofar_flux", "LOFAR_flux", "flux"]
    for c in candidates:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            return c
    # fallback: any numeric column containing "flux"
    for c in df.columns:
        if "flux" in c.lower() and pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise ValueError("Could not find a flux-like numeric column in merged PKL.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tab-emb", default="Data/tabular_embeddings.npy")
    ap.add_argument("--tab-idx", default="Data/tabular_index.csv")
    ap.add_argument("--img-emb", default="Data/dinov2_embeddings.npy")
    ap.add_argument("--img-idx", default="Data/dinov2_index.csv")

    ap.add_argument("--mlp-model", default="Data/mlp_model.pt")
    ap.add_argument("--scaler", default="Data/feature_scaler.joblib")

    ap.add_argument("--image-root", default="Data/Images")
    ap.add_argument("--device", default="cpu")

    ap.add_argument("--merged", default=None, help="Merged PKL for true flux lookup (e.g., Data/MergedDataset/merged_master_tabular_0p5.pkl)")
    ap.add_argument("--target", default=None, help="Target/flux column name in merged PKL (optional)")

    ap.add_argument("--seed", type=int, default=None, help="If omitted, uses true randomness (different galaxy each run).")
    ap.add_argument("--outdir", default="Data/demo_outputs")
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)  # None => random each run

    X_tab, mids_tab = load_embeddings(Path(args.tab_emb), Path(args.tab_idx))
    X_img, mids_img = load_embeddings(Path(args.img_emb), Path(args.img_idx))

    tab_map = {int(mid): i for i, mid in enumerate(mids_tab)}
    img_map = {int(mid): i for i, mid in enumerate(mids_img)}

    image_root = Path(args.image_root)

    common = set(tab_map.keys()).intersection(img_map.keys())
    candidates = [mid for mid in common if (image_root / f"{mid:06d}.jpg").exists()]
    if not candidates:
        raise SystemExit("No overlapping master_index that also has an image file.")

    mid = int(rng.choice(candidates))

    # Build fused feature vector
    x = np.concatenate([X_tab[tab_map[mid]], X_img[img_map[mid]]], axis=0).reshape(1, -1).astype(np.float32)

    # Standardize features (IMPORTANT)
    scaler = joblib.load(Path(args.scaler))
    x_s = scaler.transform(x)

    # Load MLP checkpoint
    ckpt = torch.load(Path(args.mlp_model), map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    in_dim = int(ckpt.get("in_dim", x_s.shape[1]))
    hidden_dims = tuple(ckpt.get("hidden_dims", [256, 128]))
    dropout = float(ckpt.get("dropout", 0.0))
    transform = str(ckpt.get("transform", "log1p"))
    y_offset = float(ckpt.get("y_offset", 0.0))

    dev = torch.device(args.device)
    model = MLP(in_dim, hidden_dims=hidden_dims, dropout=dropout).to(dev)
    model.load_state_dict(state, strict=False)
    model.eval()

    with torch.no_grad():
        yhat_t = float(model(torch.from_numpy(x_s).to(dev)).cpu().numpy().reshape(-1)[0])

    yhat_raw = invert_to_raw(yhat_t, transform, y_offset)

    # True flux lookup (optional)
    y_true_raw = None
    target_col = None
    if args.merged:
        dfm = pd.read_pickle(args.merged)
        if "master_index" not in dfm.columns:
            raise ValueError("Merged PKL is missing 'master_index' column.")
        target_col = args.target or pick_flux_column(dfm)
        row = dfm.loc[dfm["master_index"].astype(int) == mid]
        if len(row) > 0:
            y_true_raw = float(pd.to_numeric(row.iloc[0][target_col], errors="coerce"))
            if not np.isfinite(y_true_raw):
                y_true_raw = None

    # ONE print line (now includes true if available)
    if y_true_raw is None:
        print(f"master_index={mid} predicted_flux={yhat_raw:.6f}")
    else:
        print(f"master_index={mid} true_flux={y_true_raw:.6f} predicted_flux={yhat_raw:.6f}")

    # ONE output image (with both numbers)
    img_path = image_root / f"{mid:06d}.jpg"
    img = Image.open(img_path).convert("RGB")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_png = outdir / f"demo_{mid:06d}.png"

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img)
    ax1.axis("off")
    ax1.set_title(f"master_index={mid}")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis("off")

    if y_true_raw is None:
        txt = f"Predicted flux:\n{yhat_raw:.6f}"
    else:
        txt = f"True flux:\n{y_true_raw:.6f}\n\nPredicted flux:\n{yhat_raw:.6f}"

    ax2.text(0.05, 0.55, txt, fontsize=20, family="monospace", va="center")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    if args.open:
        import subprocess
        subprocess.run(["open", str(out_png)])


if __name__ == "__main__":
    main()
