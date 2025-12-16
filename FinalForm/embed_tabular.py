import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KDTree
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

#Curated tabular feature sets (NO radio columns)

KEY_FEATURES_FULL = [
    "mag_g", "mag_r", "mag_z", "mag_w1", "mag_w2",
    "magerr_g", "magerr_r", "magerr_z", "magerr_w1", "magerr_w2",
    "optRA", "optDec"
]

KEY_FEATURES_ENRICHED = [
    "z_best", "zphot", "zphot_err", "flag_qual",
    "mag_g", "mag_r", "mag_z", "mag_w1", "mag_w2",
    "magerr_g", "magerr_r", "magerr_z", "magerr_w1", "magerr_w2",
    "g_rest", "r_rest", "z_rest", "U_rest", "V_rest", "J_rest", "K_rest",
    "w1_rest", "w2_rest",
    "Mass_median", "Mass_l68", "Mass_u68",
    "r_50", "r_50_err",
    "pstar",
    "optRA", "optDec"
]

#Paths / defaults

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Data"
MERGED_DIR = DATA_DIR / "MergedDataset"

MASTER_CSV = DATA_DIR / "gz_decals_gz2_master.csv"
UNIQUE_MASTER_PKL = DATA_DIR / "unique_master_indices.pkl"

DEFAULT_MATCH_TOL_ARCSEC = 0.5

# By default we avoid embedding likely targets/leakage (kept for compatibility)
DEFAULT_EXCLUDE_SUBSTRINGS = [
    "flux", "lum", "power", "radio", "s_int", "s_peak", "fint", "fpeak"
]

#Helpers

def choose_device(cli_device: str | None) -> torch.device:
    if cli_device:
        return torch.device(cli_device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _first_col_as_array(df: pd.DataFrame, col: str) -> np.ndarray:
    sub = df.loc[:, col]
    if isinstance(sub, pd.DataFrame):
        return sub.iloc[:, 0].to_numpy()
    return sub.to_numpy()


def _spherical_to_xyz(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(ra_deg.astype(float))
    dec = np.deg2rad(dec_deg.astype(float))
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.vstack((x, y, z)).T


def _pick_merged_pkl(cli_path: str | None) -> Path:
    if cli_path:
        p = Path(cli_path)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"Merged PKL not found: {p}")
        return p

    if not MERGED_DIR.exists():
        raise FileNotFoundError(f"Merged dataset directory not found: {MERGED_DIR}")

    candidates = sorted(MERGED_DIR.glob("merged_master_tabular_*.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No merged_master_tabular_*.pkl found in: {MERGED_DIR}")

    # Prefer 0p1 if present, else first sorted
    for p in candidates:
        if p.name.endswith("_0p1.pkl"):
            return p
    return candidates[0]


def _normalize_master_index_after_reset(df: pd.DataFrame) -> pd.DataFrame:
    if "master_index" in df.columns:
        return df
    for cand in ("index", "level_0", "Unnamed: 0"):
        if cand in df.columns:
            return df.rename(columns={cand: "master_index"})
    return df


def ensure_master_index(df: pd.DataFrame, master_csv_path: Path, match_tol_arcsec: float) -> pd.DataFrame:
    if "master_index" in df.columns:
        return df

    if "Unnamed: 0" in df.columns:
        return df.rename(columns={"Unnamed: 0": "master_index"})

    # Need sky coords in df
    if ("optRA" not in df.columns) or ("optDec" not in df.columns):
        if "ra" in df.columns and "dec" in df.columns:
            df = df.rename(columns={"ra": "optRA", "dec": "optDec"})
        else:
            raise ValueError("Merged dataframe missing 'master_index' and also missing optRA/optDec (or ra/dec).")

    if not master_csv_path.exists():
        raise FileNotFoundError(f"Master CSV not found (needed to reconstruct master_index): {master_csv_path}")

    master = pd.read_csv(master_csv_path)

    # master_index might be saved as Unnamed: 0
    if "master_index" not in master.columns and "Unnamed: 0" in master.columns:
        master = master.rename(columns={"Unnamed: 0": "master_index"})
    if "master_index" not in master.columns:
        # last resort: use row number
        master.insert(0, "master_index", np.arange(len(master), dtype=int))
        print("[warn] Master CSV has no 'master_index' column; using row number as master_index.")

    # master sky coords
    if "ra" in master.columns and "dec" in master.columns:
        m_ra = master["ra"].to_numpy()
        m_dec = master["dec"].to_numpy()
    elif "optRA" in master.columns and "optDec" in master.columns:
        m_ra = master["optRA"].to_numpy()
        m_dec = master["optDec"].to_numpy()
    else:
        raise ValueError("Master CSV must have ra/dec or optRA/optDec columns.")

    df_ra = _first_col_as_array(df, "optRA")
    df_dec = _first_col_as_array(df, "optDec")
    keep_df = np.isfinite(df_ra) & np.isfinite(df_dec)

    keep_m = np.isfinite(m_ra) & np.isfinite(m_dec)
    master_valid = master.loc[keep_m].reset_index(drop=True)

    xyz_m = _spherical_to_xyz(m_ra[keep_m], m_dec[keep_m])
    xyz_df = _spherical_to_xyz(df_ra[keep_df], df_dec[keep_df])

    tree = KDTree(xyz_m)
    dist, nn_idx = tree.query(xyz_df, k=1)
    dist = dist.flatten()
    nn_idx = nn_idx.flatten()

    # KDTree on xyz returns chord distance; convert arcsec -> chord threshold
    radius_rad = np.deg2rad(match_tol_arcsec / 3600.0)
    chord_thresh = 2.0 * np.sin(radius_rad / 2.0)
    good = dist <= chord_thresh

    if good.sum() == 0:
        raise RuntimeError(f"Failed to reconstruct master_index: no matches within {match_tol_arcsec} arcsec.")

    df_valid = df.loc[keep_df].reset_index(drop=True)
    df_good = df_valid.loc[good].copy()

    matched_master_index = master_valid.loc[nn_idx[good], "master_index"].to_numpy(dtype=int)
    df_good.insert(0, "master_index", matched_master_index)

    print(f"[warn] merged PKL had no master_index; reconstructed {len(df_good)} matches (<= {match_tol_arcsec} arcsec).")
    return df_good

#Autoencoder

class TabularAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z


def train_autoencoder(X: np.ndarray, latent_dim: int, epochs: int, batch: int, lr: float, device: torch.device) -> np.ndarray:
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    loader = DataLoader(TensorDataset(X_t), batch_size=batch, shuffle=True, drop_last=False)

    model = TabularAE(X.shape[1], latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for ep in range(1, epochs + 1):
        last = None
        for (xb,) in loader:
            recon, _ = model(xb)
            loss = loss_fn(recon, xb)

            # fail fast if something goes non-finite
            if not torch.isfinite(loss):
                raise RuntimeError("Autoencoder loss became non-finite (NaN/Inf). Check preprocessing.")

            opt.zero_grad()
            loss.backward()
            opt.step()
            last = float(loss.item())

        if last is None:
            last = float("nan")
        print(f"Epoch {ep:03d}/{epochs}  Loss={last:.6f}")

    model.eval()
    with torch.no_grad():
        _, Z = model(X_t)
        return Z.detach().cpu().numpy().astype(np.float32)

# CLI

def parse_args():
    p = argparse.ArgumentParser(description="Embed curated numeric tabular columns with a small autoencoder.")
    p.add_argument("--merged", type=str, default=None, help="Path to merged_master_tabular_*.pkl")
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default=None, help="torch device string, e.g. cpu/cuda/mps")
    p.add_argument("--match-tol-arcsec", type=float, default=DEFAULT_MATCH_TOL_ARCSEC,
                   help="Only used if master_index must be reconstructed from optRA/optDec.")
    p.add_argument("--include-targets", action="store_true",
                   help="(unused here) kept for compatibility with older versions.")
    p.add_argument(
        "--feature-set",
        choices=["full", "enriched"],
        default="enriched",
        help="Which curated tabular feature set to embed."
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = choose_device(args.device)
    print(f"Device: {device.type}")

    merged_pkl = _pick_merged_pkl(args.merged)
    print(f"[info] merged PKL: {merged_pkl}")

    if not UNIQUE_MASTER_PKL.exists():
        raise FileNotFoundError(f"unique_master_indices.pkl not found: {UNIQUE_MASTER_PKL}")
    unique_master = np.asarray(pd.read_pickle(UNIQUE_MASTER_PKL), dtype=int)
    print(f"Loaded unique_master_indices: {len(unique_master)}")

    df = pd.read_pickle(merged_pkl)

    # Drop duplicate column names (merge can create duplicated labels)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Ensure master_index exists (may filter rows)
    df = ensure_master_index(df, MASTER_CSV, match_tol_arcsec=float(args.match_tol_arcsec))

    # Align to unique_master ordering
    df = df[df["master_index"].isin(unique_master)].copy()
    df = df.set_index("master_index")
    df.index.name = "master_index"  # force index name so reset_index yields correct column
    df = df.reindex(pd.Index(unique_master, name="master_index"))
    df = df.dropna(how="all")
    df = df.reset_index()

    df = _normalize_master_index_after_reset(df)
    if "master_index" not in df.columns:
        raise ValueError(f"master_index missing after alignment. Columns: {list(df.columns)[:50]}")

    kept_master = pd.to_numeric(df["master_index"], errors="coerce").astype(int).to_numpy()
    print(f"Tabular rows aligned & kept: {len(df)} (out of {len(unique_master)})")

    # Select curated feature columns ONLY
    if args.feature_set == "full":
        feature_cols = list(KEY_FEATURES_FULL)
    else:
        feature_cols = list(KEY_FEATURES_ENRICHED)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns for --feature-set {args.feature_set}: {missing}\n"
            f"Available columns (first 60): {list(df.columns)[:60]}"
        )

    print(f"Embedding curated feature columns ({args.feature_set}): {len(feature_cols)}")

    Xdf = df[feature_cols].copy()
    for c in feature_cols:
        Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")

    # Impute + standardize (finite-safe)
    Xnp = Xdf.to_numpy(dtype=np.float32)

    # turn inf/-inf into nan so they get imputed
    Xnp[~np.isfinite(Xnp)] = np.nan

    # nanmedian per column; if a column is all-nan, median becomes nan -> replace with 0
    med = np.nanmedian(Xnp, axis=0)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)

    # impute: replace nan with column median
    nan_mask = ~np.isfinite(Xnp)
    if nan_mask.any():
        Xnp[nan_mask] = np.take(med, np.where(nan_mask)[1])

    # compute mean/std AFTER imputation
    mean = Xnp.mean(axis=0).astype(np.float32)
    std = Xnp.std(axis=0).astype(np.float32)

    # guard against zero/near-zero/invalid std
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0).astype(np.float32)

    X = ((Xnp - mean) / std).astype(np.float32)

    # hard stop if anything is still non-finite
    if not np.isfinite(X).all():
        bad_cols = [feature_cols[i] for i in np.where(~np.isfinite(X).any(axis=0))[0]]
        raise ValueError(f"Non-finite values remain after preprocessing in columns: {bad_cols}")

    scaler = {
        "median": {feature_cols[i]: float(med[i]) for i in range(len(feature_cols))},
        "mean":   {feature_cols[i]: float(mean[i]) for i in range(len(feature_cols))},
        "std":    {feature_cols[i]: float(std[i]) for i in range(len(feature_cols))},
    }

    print("Training tabular autoencoder...")
    Z = train_autoencoder(X, args.latent_dim, args.epochs, args.batch, args.lr, device)

    # Save outputs
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_emb = DATA_DIR / "tabular_embeddings.npy"
    out_idx = DATA_DIR / "tabular_index.csv"
    out_feat = DATA_DIR / "tabular_features.json"
    out_scaler = DATA_DIR / "tabular_scaler.json"

    np.save(out_emb, Z)
    pd.DataFrame(
        {"row_id": np.arange(len(kept_master), dtype=int), "master_index": kept_master}
    ).to_csv(out_idx, index=False)

    out_feat.write_text(json.dumps(feature_cols, indent=2))
    out_scaler.write_text(json.dumps(scaler, indent=2))

    print("Saved:", out_emb)
    print("Saved:", out_idx)
    print("Saved:", out_feat)
    print("Saved:", out_scaler)
    print("DONE.")


if __name__ == "__main__":
    main()
