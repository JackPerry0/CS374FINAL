import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

#Paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Data"
MERGED_DIR = DATA_DIR / "MergedDataset"
MASTER_CSV = DATA_DIR / "gz_decals_gz2_master.csv"

IMG_EMB = DATA_DIR / "dinov2_embeddings.npy"
IMG_IDX = DATA_DIR / "dinov2_index.csv"

TAB_EMB = DATA_DIR / "tabular_embeddings.npy"
TAB_IDX = DATA_DIR / "tabular_index.csv"

#Device / reproducibility
def choose_device(cli_device: str | None) -> torch.device:
    if cli_device:
        return torch.device(cli_device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

#Robust master_index reconstruction (same idea as embed_tabular.py)
def _first_col_as_array(df: pd.DataFrame, col: str) -> np.ndarray:
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


def ensure_master_index(df: pd.DataFrame, match_tol_arcsec: float) -> pd.DataFrame:
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

#Merged PKL picking + target selection
def pick_merged(cli_path: str | None) -> Path:
    if cli_path:
        p = Path(cli_path)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    candidates = sorted(MERGED_DIR.glob("merged_master_tabular_*.pkl"))
    if not candidates:
        candidates = sorted(Path(".").glob("merged_master_tabular_*.pkl"))
    if not candidates:
        raise FileNotFoundError("No merged_master_tabular_*.pkl found.")
    # prefer 0p1
    candidates.sort(key=lambda p: (0 if "0p1" in p.name else 1, p.name))
    return candidates[0]


def auto_pick_target(df: pd.DataFrame) -> str:
    # Heuristic: find numeric column that looks like flux, but avoid obvious non-targets
    bad = ("ra", "dec", "err", "sigma")
    good = ("flux", "fint", "s_int", "total", "int", "peak")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    candidates = []
    for c in num_cols:
        cl = c.lower()
        if any(b in cl for b in bad):
            continue
        if any(g in cl for g in good):
            candidates.append(c)
    if not candidates:
        raise RuntimeError("Could not auto-pick target column. Pass --target explicitly.")
    # Prefer 'total'/'int' over 'peak'
    candidates.sort(key=lambda c: (0 if "total" in c.lower() or "int" in c.lower() else 1, c))
    return candidates[0]

#Embedding loaders
def read_index_csv(idx_path: Path) -> pd.DataFrame:
    idx = pd.read_csv(idx_path)
    if "master_index" not in idx.columns:
        raise ValueError(f"index csv missing master_index: {idx_path}")
    return idx


def load_embeddings_with_index(emb_path: Path, idx_path: Path, name: str):
    if not emb_path.exists() or not idx_path.exists():
        raise FileNotFoundError(f"Missing {name} files: {emb_path} and/or {idx_path}")

    Z = np.load(emb_path)
    idx = read_index_csv(idx_path)

    if len(idx) != Z.shape[0]:
        k = min(len(idx), Z.shape[0])
        print(f"[warn] {name} index/emb mismatch: idx={len(idx)} emb={Z.shape[0]} -> trimming to {k}")
        idx = idx.iloc[:k].reset_index(drop=True)
        Z = Z[:k]

    mids = idx["master_index"].astype(int).to_numpy()
    row = np.arange(len(mids), dtype=int)
    map_df = pd.DataFrame({"master_index": mids, f"{name}_row": row})
    return Z.astype(np.float32, copy=False), map_df

#Target transform
def transform_y(y: np.ndarray, mode: str):
    y = y.astype(float)
    offset = 0.0
    if mode == "none":
        return y, offset
    if mode in ("log1p", "log10p1"):
        miny = float(np.nanmin(y))
        if miny < 0.0:
            offset = -miny
        y2 = y + offset
        if mode == "log1p":
            return np.log1p(y2), offset
        return np.log10(1.0 + y2), offset
    raise ValueError(mode)


def invert_y(y_t: np.ndarray, mode: str, offset: float):
    if mode == "none":
        return y_t
    if mode == "log1p":
        return np.expm1(y_t) - offset
    if mode == "log10p1":
        return (10.0 ** y_t - 1.0) - offset
    raise ValueError(mode)

#Models (Ridge + MLP)
class MLP(nn.Module):
    def __init__(self, d_in: int, hidden=(256, 128), dropout: float = 0.0):
        super().__init__()
        layers = []
        prev = d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout and dropout > 0:
                layers += [nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_train: np.ndarray, y_train: np.ndarray, seed: int,
              hidden=(256, 128), dropout=0.0, epochs=80, batch=256, lr=1e-3,
              weight_decay=1e-5, patience=10, device=torch.device("cpu")):
    # simple train/val split from TRAIN only
    n = len(X_train)
    idx = np.arange(n)
    tr, va = train_test_split(idx, test_size=0.2, random_state=seed)

    Xtr = torch.tensor(X_train[tr], dtype=torch.float32)
    ytr = torch.tensor(y_train[tr], dtype=torch.float32)
    Xva = torch.tensor(X_train[va], dtype=torch.float32)
    yva = torch.tensor(y_train[va], dtype=torch.float32)

    tr_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch, shuffle=True, drop_last=False)

    model = MLP(X_train.shape[1], hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best = float("inf")
    best_state = None
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred_va = model(Xva.to(device))
            val_loss = float(loss_fn(pred_va, yva.to(device)).cpu().item())

        if val_loss < best - 1e-6:
            best = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

#Calling embed_tabular for leakage-safe embeddings
def run_embed_tabular_trainfit(embed_script: Path,
                              merged_pkl: Path,
                              train_ids_csv: Path,
                              out_prefix: Path,
                              latent_dim: int, epochs: int, batch: int, lr: float,
                              device: str | None,
                              match_tol_arcsec: float,
                              tab_feature_set: str):
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    train_ids_csv.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(embed_script),
        "--merged", str(merged_pkl),
        "--train-ids", str(train_ids_csv),
        "--out-prefix", str(out_prefix),
        "--latent-dim", str(latent_dim),
        "--epochs", str(epochs),
        "--batch", str(batch),
        "--lr", str(lr),
        "--match-tol-arcsec", str(match_tol_arcsec),
        "--save-model",
    ]
    cmd += ["--feature-set", tab_feature_set]
    if device:
        cmd += ["--device", device]

    print("[info] running tabular embedding (trainfit):")
    print("       " + " ".join(cmd))
    subprocess.run(cmd, check=True)

#CLI
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--merged", type=str, default=None)
    p.add_argument("--target", type=str, default=None)
    p.add_argument("--use", choices=["img", "tab", "both"], default="both")
    p.add_argument("--transform", choices=["none", "log1p", "log10p1"], default="log1p")
    p.add_argument("--match-tol-arcsec", type=float, default=0.5)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=123)

    # Ridge
    p.add_argument("--ridge-alpha", type=float, default=1.0)

    # MLP
    p.add_argument("--mlp-hidden", type=str, default="256,128")
    p.add_argument("--mlp-dropout", type=float, default=0.0)
    p.add_argument("--mlp-epochs", type=int, default=80)
    p.add_argument("--mlp-batch", type=int, default=256)
    p.add_argument("--mlp-lr", type=float, default=1e-3)
    p.add_argument("--mlp-weight-decay", type=float, default=1e-5)
    p.add_argument("--mlp-patience", type=int, default=10)

    # Tabular embedding mode (leakage-safe default)
    p.add_argument("--tab-embed-mode", choices=["precomputed", "trainfit"], default="trainfit",
                   help="precomputed=use Data/tabular_embeddings.npy; trainfit=fit AE on TRAIN ONLY by calling embed_tabular.py")
    p.add_argument("--tab-out-prefix", type=str, default=str(DATA_DIR / "tabular_trainfit"),
                   help="Where to write trainfit tabular outputs. Prefix for *_embeddings.npy, *_index.csv, etc.")
    p.add_argument("--tab-latent-dim", type=int, default=32)
    p.add_argument("--tab-ae-epochs", type=int, default=50)
    p.add_argument("--tab-ae-batch", type=int, default=128)
    p.add_argument("--tab-ae-lr", type=float, default=1e-3)
    p.add_argument("--reuse-tab-embeddings", action="store_true",
                   help="If set, and trainfit outputs already exist, do not re-run embed_tabular.py.")

    p.add_argument("--device", type=str, default=None)

    p.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True,
                help="Save diagnostic plots (default: enabled). Use --no-plots to disable.")
    p.add_argument("--plot-dir", type=str, default=str(DATA_DIR / "plots"),
                help="Directory to write plots.")
    p.add_argument("--show-plots", action="store_true",
                help="Also display plots interactively (may not work on headless systems).")


    # save paths
    p.add_argument("--ridge-out", type=str, default=str(DATA_DIR / "ridge_model.joblib"))
    p.add_argument("--mlp-out", type=str, default=str(DATA_DIR / "mlp_model.pt"))
    p.add_argument("--metrics-out", type=str, default=str(DATA_DIR / "flux_model_metrics.json"))
    p.add_argument("--preds-out", type=str, default=str(DATA_DIR / "flux_model_predictions.csv"))
    p.add_argument("--scaler-out", type=str, default=str(DATA_DIR / "feature_scaler.joblib"))
    p.add_argument("--tab-feature-set", choices=["full", "enriched"], default="enriched",
               help="Which curated tabular feature set to embed (passed to embed_tabular script).")

    return p.parse_args()

def save_4panel_diagnostics(out, y_pred_col, plot_path, title_prefix="RAW (flux)"):
    y_true = out["y_true"].to_numpy(dtype=float)
    y_pred = out[y_pred_col].to_numpy(dtype=float)
    resid = y_pred - y_true

    # Stats for header
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan")
    mae = float(mean_absolute_error(y_true, y_pred))
    medae = float(np.median(np.abs(resid)))
    std = float(np.std(resid))

    # Robust axis limits (ignore extreme tails)
    r_lo, r_hi = np.percentile(resid, [0.5, 99.5])
    y_lo, y_hi = np.percentile(y_true, [0.5, 99.5])
    p_lo, p_hi = np.percentile(y_pred, [0.5, 99.5])
    f_lo = min(y_lo, p_lo)
    f_hi = max(y_hi, p_hi)

    fig, axs = plt.subplots(2, 2, figsize=(14, 8), dpi=200)

    fig.suptitle(
        f"{title_prefix} - {y_pred_col}\n"
        f"Corr={corr:.4f}, MAE={mae:.3f}, MedAE={medae:.3f}, Std={std:.3f}",
        fontsize=14
    )

    # (1) True vs Predicted
    ax = axs[0, 0]
    ax.scatter(y_true, y_pred, s=8, alpha=0.35)
    ax.set_title("True vs Predicted", fontsize=12)
    ax.set_xlabel("True Flux", fontsize=11)
    ax.set_ylabel("Predicted Flux", fontsize=11)
    ax.set_xlim(f_lo, f_hi)
    ax.set_ylim(f_lo, f_hi)

    # Optional: y=x line (helps legibility)
    ax.plot([f_lo, f_hi], [f_lo, f_hi], linestyle="--", linewidth=1)

    # (2) Residual histogram (zoomed)
    ax = axs[0, 1]
    ax.hist(resid, bins=60)
    ax.set_title("Residual Distribution", fontsize=12)
    ax.set_xlabel("Residual = Pred - True", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_xlim(r_lo, r_hi)

    # (3) Residuals vs Predicted
    ax = axs[1, 0]
    ax.scatter(y_pred, resid, s=8, alpha=0.35)
    ax.axhline(y=0, linestyle="--", linewidth=1)
    ax.set_title("Residuals vs Predicted", fontsize=12)
    ax.set_xlabel("Predicted Flux", fontsize=11)
    ax.set_ylabel("Residual", fontsize=11)
    ax.set_xlim(p_lo, p_hi)
    ax.set_ylim(r_lo, r_hi)

    # (4) Flux distribution comparison (True vs Pred)
    ax = axs[1, 1]
    bins = np.linspace(f_lo, f_hi, 60)
    ax.hist(y_true, bins=bins, alpha=0.6, label="True")
    ax.hist(y_pred, bins=bins, alpha=0.6, label="Predicted")
    ax.set_title("Flux Distribution Comparison", fontsize=12)
    ax.set_xlabel("Flux", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.legend(fontsize=10)

    for ax in axs.ravel():
        ax.tick_params(labelsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    print(f"Device: {device.type}")

    merged_pkl = pick_merged(args.merged)
    df = pd.read_pickle(merged_pkl)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = ensure_master_index(df, args.match_tol_arcsec)

    # Pick target column
    target_col = args.target if args.target else auto_pick_target(df)
    if target_col not in df.columns:
        raise ValueError(f"target column not found: {target_col}")

    y = pd.to_numeric(df[target_col], errors="coerce")
    tgt = pd.DataFrame({"master_index": df["master_index"].astype(int).to_numpy(), "y": y.to_numpy()})
    tgt = tgt.dropna(subset=["y"]).drop_duplicates("master_index").copy()

    # Load image embeddings (pretrained; safe to precompute)
    img_Z, img_map = None, None
    if args.use in ("img", "both"):
        img_Z, img_map = load_embeddings_with_index(IMG_EMB, IMG_IDX, "img")
        print(f"[info] img embeddings: {img_Z.shape}")

    # Build a preliminary key (target + required non-tab modalities)
    key0 = tgt
    if img_map is not None:
        key0 = key0.merge(img_map, on="master_index", how="inner")

    if len(key0) < 200:
        raise RuntimeError("Too few matched rows after target/img intersection. Check inputs.")

    # Split BEFORE any tabular AE fitting
    idx_all = np.arange(len(key0), dtype=int)
    idx_train0, idx_test0 = train_test_split(idx_all, test_size=args.test_size, random_state=args.seed)
    train_ids = set(key0.iloc[idx_train0]["master_index"].astype(int).to_list())
    test_ids = set(key0.iloc[idx_test0]["master_index"].astype(int).to_list())

    # Prepare / load tabular embeddings
    tab_Z, tab_map = None, None
    if args.use in ("tab", "both"):
        if args.tab_embed_mode == "precomputed":
            tab_Z, tab_map = load_embeddings_with_index(TAB_EMB, TAB_IDX, "tab")
            print(f"[info] tab embeddings (precomputed): {tab_Z.shape}")
        else:
            # trainfit: run embed_tabular.py using TRAIN IDs only
            embed_script = ROOT / "embed_tabular_leakfree.py"
            out_prefix = Path(args.tab_out_prefix)
            emb_path = Path(str(out_prefix) + "_embeddings.npy")
            idx_path = Path(str(out_prefix) + "_index.csv")

            if args.reuse_tab_embeddings and emb_path.exists() and idx_path.exists():
                print(f"[info] reusing existing trainfit tabular embeddings: {emb_path}")
            else:
                train_ids_csv = DATA_DIR / f"train_ids_seed{args.seed}.csv"
                pd.DataFrame({"master_index": sorted(train_ids)}).to_csv(train_ids_csv, index=False)

                run_embed_tabular_trainfit(
                    embed_script=embed_script,
                    merged_pkl=merged_pkl,
                    train_ids_csv=train_ids_csv,
                    out_prefix=out_prefix,
                    latent_dim=args.tab_latent_dim,
                    epochs=args.tab_ae_epochs,
                    batch=args.tab_ae_batch,
                    lr=args.tab_ae_lr,
                    device=args.device,
                    match_tol_arcsec=args.match_tol_arcsec,
                    tab_feature_set=args.tab_feature_set,
                )

            tab_Z, tab_map = load_embeddings_with_index(emb_path, idx_path, "tab")
            print(f"[info] tab embeddings (trainfit): {tab_Z.shape}")

    # Final key with all required modalities
    key = key0
    if tab_map is not None:
        key = key.merge(tab_map, on="master_index", how="inner")

    print(f"[info] rows after full intersection: {len(key)}")
    if len(key) < 200:
        raise RuntimeError("Too few matched rows after adding tab embeddings.")

    # Build train/test masks based on master_index membership from the original split
    is_train = key["master_index"].astype(int).isin(train_ids).to_numpy()
    is_test = key["master_index"].astype(int).isin(test_ids).to_numpy()

    # Sanity: every row should be in exactly one set
    if not np.all(is_train ^ is_test):
        # If this happens, something altered key0 between split and key creation.
        # Fall back to splitting again deterministically on current key.
        idx_all2 = np.arange(len(key), dtype=int)
        idx_train2, idx_test2 = train_test_split(idx_all2, test_size=args.test_size, random_state=args.seed)
        is_train = np.zeros(len(key), dtype=bool); is_train[idx_train2] = True
        is_test = ~is_train
        print("[warn] split mismatch after intersections; re-splitting on final key.")

    # Build X
    parts = []
    if img_map is not None:
        parts.append(img_Z[key["img_row"].to_numpy(dtype=int)])
    if tab_map is not None:
        parts.append(tab_Z[key["tab_row"].to_numpy(dtype=int)])
    X = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
    y_raw = key["y"].to_numpy(dtype=float)

    # Transform y
    y_t, y_offset = transform_y(y_raw, args.transform)

    # Split arrays
    X_train = X[is_train]
    X_test = X[is_test]
    y_train = y_t[is_train]
    y_test = y_t[is_test]

    # Standardize features for the predictor (fit on TRAIN only)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, args.scaler_out)

    # Ridge
    ridge = Ridge(alpha=args.ridge_alpha, random_state=args.seed)
    ridge.fit(X_train_s, y_train)
    yhat_r_t = ridge.predict(X_test_s)

    # MLP
    hidden = tuple(int(x) for x in args.mlp_hidden.split(",") if x.strip())
    mlp = train_mlp(
        X_train_s, y_train, seed=args.seed,
        hidden=hidden, dropout=args.mlp_dropout,
        epochs=args.mlp_epochs, batch=args.mlp_batch, lr=args.mlp_lr,
        weight_decay=args.mlp_weight_decay, patience=args.mlp_patience,
        device=device,
    )
    mlp.eval()
    with torch.no_grad():
        yhat_m_t = mlp(torch.tensor(X_test_s, dtype=torch.float32, device=device)).detach().cpu().numpy()

    # Invert to raw flux space for reporting
    y_test_raw = invert_y(y_test, args.transform, y_offset)
    yhat_r_raw = invert_y(yhat_r_t, args.transform, y_offset)
    yhat_m_raw = invert_y(yhat_m_t, args.transform, y_offset)

    metrics = {
        "target_col": target_col,
        "use": args.use,
        "transform": args.transform,
        "n_total": int(len(key)),
        "n_train": int(is_train.sum()),
        "n_test": int(is_test.sum()),
        "ridge_alpha": args.ridge_alpha,
        "ridge_mae": float(mean_absolute_error(y_test_raw, yhat_r_raw)),
        "ridge_r2": float(r2_score(y_test_raw, yhat_r_raw)),
        "mlp_hidden": list(hidden),
        "mlp_dropout": args.mlp_dropout,
        "mlp_mae": float(mean_absolute_error(y_test_raw, yhat_m_raw)),
        "mlp_r2": float(r2_score(y_test_raw, yhat_m_raw)),
        "tab_embed_mode": args.tab_embed_mode if args.use in ("tab", "both") else None,
        "tab_out_prefix": args.tab_out_prefix if args.tab_embed_mode == "trainfit" else None,
        "seed": args.seed,
        "test_size": args.test_size,
        "tab_feature_set": args.tab_feature_set if args.use in ("tab","both") else None,

    }

    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2))
    joblib.dump(ridge, args.ridge_out)
    torch.save(mlp.state_dict(), args.mlp_out)

    # Save predictions (master_index aligned)
    out = pd.DataFrame({
        "master_index": key.loc[is_test, "master_index"].astype(int).to_numpy(),
        "y_true": y_test_raw,
        "y_pred_ridge": yhat_r_raw,
        "y_pred_mlp": yhat_m_raw,
    })
    out.to_csv(args.preds_out, index=False)
    
    if args.plots:
        plot_dir = plot_dir = DATA_DIR / "leakfreeplots"
        save_4panel_diagnostics(out, "y_pred_mlp",  plot_dir / "mlp_diagnostics.png")
        save_4panel_diagnostics(out, "y_pred_ridge", plot_dir / "ridge_diagnostics.png")
        print(f"[done] saved plots to: {plot_dir}")


    print("[done] wrote:")
    print(f"  metrics: {args.metrics_out}")
    print(f"  preds  : {args.preds_out}")
    print(f"  ridge  : {args.ridge_out}")
    print(f"  mlp    : {args.mlp_out}")


if __name__ == "__main__":
    main()
