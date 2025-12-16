#!/usr/bin/env python3
"""
embed_tabular.py (leakage-safe mode added)

Embeds numeric tabular columns from a merged optical+LOFAR dataset using a small
autoencoder, aligned to Data/unique_master_indices.pkl ordering.

NEW (leakage-safe option):
  --train-ids <path>  Fit imputer/scaler + autoencoder ONLY on the provided train IDs,
                      then embed ALL rows using the frozen train-fitted transform.

Outputs (written to Data/ by default via --out-prefix):
  - {out_prefix}_embeddings.npy        (N, latent_dim)
  - {out_prefix}_index.csv             row_id, master_index
  - {out_prefix}_features.json         list[str] embedded feature columns
  - {out_prefix}_scaler.json           train-fitted median/mean/std
  - {out_prefix}_ae.pt                 (optional) trained AE weights

Run (from project root, e.g. FinalForm/):
  python embed_tabular.py --merged Data/MergedDataset/merged_master_tabular_0p1.pkl

Leakage-safe (fit only on train IDs):
  python embed_tabular.py --merged ... --train-ids Data/train_ids.csv

Notes:
- By default we EXCLUDE target-ish columns (e.g., anything containing 'flux') from the
  embedded features, unless you pass --include-targets.
- If your merged PKL lacks master_index, we reconstruct it by sky-coordinate matching
  to Data/gz_decals_gz2_master.csv.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------
# Curated tabular feature sets (NO radio columns)
# -----------------------------
KEY_FEATURES_FULL = [
    "mag_g","mag_r","mag_z","mag_w1","mag_w2",
    "magerr_g","magerr_r","magerr_z","magerr_w1","magerr_w2",
    "optRA","optDec"
]

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


# -----------------------------
# Paths / defaults
# -----------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Data"
MERGED_DIR = DATA_DIR / "MergedDataset"
MASTER_CSV = DATA_DIR / "gz_decals_gz2_master.csv"
UNIQUE_IDS_PKL = DATA_DIR / "unique_master_indices.pkl"

DEFAULT_MATCH_TOL_ARCSEC = 0.5

DEFAULT_EXCLUDE_SUBSTRINGS = [
    "flux", "lum", "power", "radio", "s_int", "s_peak", "fint", "fpeak"
]


# -----------------------------
# Device helpers
# -----------------------------
def choose_device(cli_device: str | None) -> torch.device:
    if cli_device:
        return torch.device(cli_device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -----------------------------
# Utility helpers
# -----------------------------
def _first_col_as_array(df: pd.DataFrame, col: str) -> np.ndarray:
    """If df[col] is a DataFrame due to duplicate columns, take the first."""
    x = df[col]
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    return x.to_numpy()


def _spherical_to_xyz(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(ra_deg.astype(float))
    dec = np.deg2rad(dec_deg.astype(float))
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.vstack([x, y, z]).T


def _pick_merged_pkl(cli_path: str | None) -> Path:
    if cli_path:
        p = Path(cli_path)
        if not p.exists():
            raise FileNotFoundError(f"merged pkl not found: {p}")
        return p

    # Prefer a 0p1 if present; else any merged_master_tabular*.pkl
    candidates = sorted(MERGED_DIR.glob("merged_master_tabular_*.pkl"))
    if not candidates:
        # fall back to current directory
        candidates = sorted(Path(".").glob("merged_master_tabular_*.pkl"))
    if not candidates:
        raise FileNotFoundError(
            f"No merged_master_tabular_*.pkl found in {MERGED_DIR} or current directory."
        )

    def score(p: Path) -> tuple[int, str]:
        s = p.name
        # put 0p1 first if possible
        return (0 if "0p1" in s else 1, s)

    return sorted(candidates, key=score)[0]


def _normalize_master_index_after_reset(df: pd.DataFrame) -> pd.DataFrame:
    if "master_index" in df.columns:
        return df
    for alt in ("index", "level_0"):
        if alt in df.columns:
            return df.rename(columns={alt: "master_index"})
    raise KeyError("Could not find master_index after reindex/reset_index")


def ensure_master_index(df: pd.DataFrame, match_tol_arcsec: float) -> pd.DataFrame:
    """
    Ensure df has an integer master_index column.
    If missing, reconstruct by nearest-neighbor matching on sky coords to MASTER_CSV.
    Returns a (possibly filtered) dataframe with master_index present.
    """
    if "master_index" in df.columns:
        df["master_index"] = pd.to_numeric(df["master_index"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["master_index"]).copy()
        df["master_index"] = df["master_index"].astype(int)
        return df

    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "master_index"}).copy()
        df["master_index"] = pd.to_numeric(df["master_index"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["master_index"]).copy()
        df["master_index"] = df["master_index"].astype(int)
        return df

    # Need sky coords
    if "optRA" in df.columns and "optDec" in df.columns:
        ra = _first_col_as_array(df, "optRA")
        dec = _first_col_as_array(df, "optDec")
    elif "ra" in df.columns and "dec" in df.columns:
        df = df.rename(columns={"ra": "optRA", "dec": "optDec"}).copy()
        ra = _first_col_as_array(df, "optRA")
        dec = _first_col_as_array(df, "optDec")
    else:
        raise ValueError("No master_index and no (optRA,optDec) or (ra,dec) to reconstruct it.")

    master = pd.read_csv(MASTER_CSV)
    if "master_index" not in master.columns:
        master = master.copy()
        master["master_index"] = np.arange(len(master), dtype=int)

    if "ra" in master.columns and "dec" in master.columns:
        m_ra = master["ra"].to_numpy(dtype=float)
        m_dec = master["dec"].to_numpy(dtype=float)
    elif "optRA" in master.columns and "optDec" in master.columns:
        m_ra = master["optRA"].to_numpy(dtype=float)
        m_dec = master["optDec"].to_numpy(dtype=float)
    else:
        raise ValueError("MASTER_CSV missing ra/dec columns.")

    mask = np.isfinite(ra) & np.isfinite(dec)
    df2 = df.loc[mask].copy()
    ra2 = ra[mask].astype(float)
    dec2 = dec[mask].astype(float)

    m_xyz = _spherical_to_xyz(m_ra, m_dec)
    q_xyz = _spherical_to_xyz(ra2, dec2)
    tree = cKDTree(m_xyz)
    dist, idx = tree.query(q_xyz, k=1)

    tol_rad = np.deg2rad(match_tol_arcsec / 3600.0)
    chord_thresh = 2.0 * np.sin(tol_rad / 2.0)

    good = dist <= chord_thresh
    if good.sum() == 0:
        raise RuntimeError(f"0 coordinate matches within {match_tol_arcsec} arcsec; cannot reconstruct master_index.")

    df2 = df2.loc[good].copy()
    df2["master_index"] = master.loc[idx[good], "master_index"].to_numpy(dtype=int)
    return df2


# -----------------------------
# Autoencoder model
# -----------------------------
class TabularAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


def _fit_imputer_scaler_on_train(X_all: np.ndarray, train_mask: np.ndarray):
    """Compute median/mean/std on TRAIN ONLY; return filled+standardized X and stats."""
    X_all = X_all.astype(np.float32, copy=False)

    X_train = X_all[train_mask]
    med = np.nanmedian(X_train, axis=0)
    # fill NaNs in both train and all using train medians
    X_filled = np.where(np.isfinite(X_all), X_all, med)

    mu = X_filled[train_mask].mean(axis=0)
    sd = X_filled[train_mask].std(axis=0)
    sd = np.where(sd > 1e-8, sd, 1.0)

    X_std = (X_filled - mu) / sd
    stats = {"median": med.tolist(), "mean": mu.tolist(), "std": sd.tolist()}
    return X_std.astype(np.float32), stats


def _train_ae_model(X_train_std: np.ndarray, latent_dim: int, epochs: int, batch: int, lr: float, device: torch.device):
    X_t = torch.tensor(X_train_std, dtype=torch.float32, device=device)
    loader = DataLoader(TensorDataset(X_t), batch_size=batch, shuffle=True, drop_last=False)

    model = TabularAE(X_train_std.shape[1], latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for ep in range(1, epochs + 1):
        last = None
        for (xb,) in loader:
            recon, _ = model(xb)
            loss = loss_fn(recon, xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last = float(loss.detach().cpu().item())
        print(f"Epoch {ep:03d}/{epochs}  Loss={last:.6f}")

    return model


def _encode_all(model: TabularAE, X_std: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    X_t = torch.tensor(X_std, dtype=torch.float32, device=device)
    with torch.no_grad():
        _, Z = model(X_t)
    return Z.detach().cpu().numpy().astype(np.float32)


# -----------------------------
# CLI
# -----------------------------
def _load_train_ids(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--train-ids not found: {p}")
    suf = p.suffix.lower()
    if suf in (".pkl", ".pickle"):
        arr = pd.read_pickle(p)
        return np.asarray(arr, dtype=int)
    if suf == ".npy":
        return np.asarray(np.load(p), dtype=int)
    if suf == ".csv":
        df = pd.read_csv(p)
        if "master_index" in df.columns:
            return df["master_index"].astype(int).to_numpy()
        # else: first column
        return df.iloc[:, 0].astype(int).to_numpy()
    # fallback: read as text list of ints
    txt = p.read_text().strip().split()
    return np.asarray([int(x) for x in txt], dtype=int)


def parse_args():
    p = argparse.ArgumentParser(description="Embed numeric tabular columns with a small autoencoder.")
    p.add_argument("--merged", type=str, default=None, help="Path to merged_master_tabular_*.pkl")
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default=None)

    p.add_argument("--match-tol-arcsec", type=float, default=DEFAULT_MATCH_TOL_ARCSEC,
                   help="Only used when master_index must be reconstructed via RA/Dec matching.")
    p.add_argument("--include-targets", action="store_true",
                   help="Include target-ish columns (e.g., flux) in the embedded features (NOT recommended).")

    # Leakage-safe mode
    p.add_argument("--train-ids", type=str, default=None,
                   help="Path to train IDs (csv/pkl/npy). If provided, fit imputer/scaler + AE on TRAIN ONLY.")

    # Output control
    p.add_argument("--out-prefix", type=str, default=str(DATA_DIR / "tabular"),
                   help="Output prefix. Writes {prefix}_embeddings.npy, {prefix}_index.csv, etc.")
    p.add_argument("--save-model", action="store_true",
                   help="Also save AE weights to {out_prefix}_ae.pt")

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
    merged_pkl = _pick_merged_pkl(args.merged)

    unique_master = pd.read_pickle(UNIQUE_IDS_PKL)
    unique_master = np.asarray(unique_master, dtype=int)

    print(f"[info] device={device.type}")
    print(f"[info] merged={merged_pkl}")
    print(f"[info] unique_master_indices={len(unique_master)}")
    print(f"[info] out_prefix={args.out_prefix}")

    df = pd.read_pickle(merged_pkl)
    # drop duplicated columns if any
    df = df.loc[:, ~df.columns.duplicated()].copy()

    df = ensure_master_index(df, args.match_tol_arcsec)

    # Align to unique_master ordering (drop rows not in unique list)
    df = df[df["master_index"].isin(unique_master)].copy()
    df = df.set_index("master_index").reindex(unique_master)
    df = df.dropna(how="all").reset_index()
    df = _normalize_master_index_after_reset(df)

    # -----------------------------
    # Select curated feature columns ONLY
    # -----------------------------
    if args.feature_set == "full":
        num_cols = list(KEY_FEATURES_FULL)
    else:
        num_cols = list(KEY_FEATURES_ENRICHED)

    # sanity: do NOT allow master_index in features
    num_cols = [c for c in num_cols if c != "master_index"]

    # ensure requested columns exist
    missing = [c for c in num_cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            "Missing required tabular columns for feature-set="
            f"{args.feature_set}: {missing}\n"
            f"Available columns include: {list(df.columns)[:40]} ..."
        )

    # Build X (all rows) – coerce to numeric
    X_all = df[num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

    # Remove master_index from features
    if "master_index" in num_cols:
        num_cols.remove("master_index")

    # Exclude target-ish columns unless requested
    if not args.include_targets:
        excl = tuple(s.lower() for s in DEFAULT_EXCLUDE_SUBSTRINGS)
        kept = []
        for c in num_cols:
            cl = c.lower()
            if any(sub in cl for sub in excl):
                continue
            kept.append(c)
        num_cols = kept

    if not num_cols:
        raise RuntimeError("No numeric feature columns found to embed after filtering.")

    # Build X (all rows)
    X_all = df[num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

    # Choose which rows are TRAIN for fitting
    if args.train_ids:
        train_ids = set(_load_train_ids(args.train_ids).astype(int).tolist())
        train_mask = df["master_index"].astype(int).isin(train_ids).to_numpy()
        if train_mask.sum() < 50:
            raise RuntimeError(f"Too few train rows ({train_mask.sum()}) from --train-ids; cannot fit AE reliably.")
        print(f"[info] leakage-safe fit: using {train_mask.sum()} rows for fit, embedding {len(df)} total rows")
    else:
        train_mask = np.ones(len(df), dtype=bool)
        print(f"[info] fit on ALL rows (transductive): {len(df)}")

    # Fit imputer/scaler on TRAIN ONLY; transform ALL
    X_std, scaler_stats = _fit_imputer_scaler_on_train(X_all, train_mask)

    # Train AE on TRAIN ONLY
    model = _train_ae_model(X_std[train_mask], args.latent_dim, args.epochs, args.batch, args.lr, device)

    # Embed ALL rows using frozen encoder
    Z_all = _encode_all(model, X_std, device)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    np.save(str(out_prefix) + "_embeddings.npy", Z_all)
    pd.DataFrame({"row_id": np.arange(len(df), dtype=int), "master_index": df["master_index"].astype(int)}).to_csv(
        str(out_prefix) + "_index.csv", index=False
    )
    Path(str(out_prefix) + "_features.json").write_text(json.dumps(num_cols, indent=2))
    Path(str(out_prefix) + "_scaler.json").write_text(json.dumps(scaler_stats, indent=2))

    if args.save_model:
        torch.save(model.state_dict(), str(out_prefix) + "_ae.pt")

    print(f"[done] wrote {str(out_prefix)}_embeddings.npy  shape={Z_all.shape}")


if __name__ == "__main__":
    main()
