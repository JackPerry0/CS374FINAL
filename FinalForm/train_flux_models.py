import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KDTree

import joblib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

#Master index reconstruction

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


def ensure_master_index(df: pd.DataFrame, master_csv_path: Path, match_tol_arcsec: float) -> pd.DataFrame:
    if "master_index" in df.columns:
        return df
    if "Unnamed: 0" in df.columns:
        return df.rename(columns={"Unnamed: 0": "master_index"})

    if ("optRA" not in df.columns) or ("optDec" not in df.columns):
        if "ra" in df.columns and "dec" in df.columns:
            df = df.rename(columns={"ra": "optRA", "dec": "optDec"})
        else:
            raise ValueError("Merged dataframe missing 'master_index' and also missing optRA/optDec (or ra/dec).")

    if not master_csv_path.exists():
        raise FileNotFoundError(f"Master CSV not found: {master_csv_path}")

    master = pd.read_csv(master_csv_path)
    if "master_index" not in master.columns and "Unnamed: 0" in master.columns:
        master = master.rename(columns={"Unnamed: 0": "master_index"})
    if "master_index" not in master.columns:
        master.insert(0, "master_index", np.arange(len(master), dtype=int))
        print("[warn] Master CSV has no 'master_index'; using row number as master_index.")

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
    dist, nn = tree.query(xyz_df, k=1)
    dist = dist.flatten()
    nn = nn.flatten()

    radius_rad = np.deg2rad(match_tol_arcsec / 3600.0)
    chord_thresh = 2.0 * np.sin(radius_rad / 2.0)
    good = dist <= chord_thresh
    if good.sum() == 0:
        raise RuntimeError(f"Failed to reconstruct master_index: no matches within {match_tol_arcsec} arcsec.")

    df_valid = df.loc[keep_df].reset_index(drop=True)
    df_good = df_valid.loc[good].copy()
    df_good.insert(0, "master_index", master_valid.loc[nn[good], "master_index"].to_numpy(dtype=int))

    print(f"[warn] reconstructed master_index for {len(df_good)} rows (<= {match_tol_arcsec} arcsec).")
    return df_good

# Embedding loaders / alignment

def read_index_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}

    if "master_index" not in df.columns:
        if "id" in cols_lower:
            df = df.rename(columns={cols_lower["id"]: "master_index"})
        elif "masterindex" in cols_lower:
            df = df.rename(columns={cols_lower["masterindex"]: "master_index"})

    if "master_index" not in df.columns:
        raise ValueError(f"{path} must contain master_index (or id). Columns: {df.columns.tolist()}")

    df["master_index"] = pd.to_numeric(df["master_index"], errors="coerce")
    df = df.dropna(subset=["master_index"]).copy()
    df["master_index"] = df["master_index"].astype(int)
    return df.reset_index(drop=True)


def load_embeddings_with_index(emb_path: Path, idx_path: Path, name: str):
    if not emb_path.exists() or not idx_path.exists():
        raise FileNotFoundError(f"Missing {name} files: {emb_path} and/or {idx_path}")

    Z = np.load(emb_path)
    idx = read_index_csv(idx_path)

    # If mismatch, trim to min (safe if one file is an overwritten partial)
    if len(idx) != Z.shape[0]:
        k = min(len(idx), Z.shape[0])
        print(f"[warn] {name} index/emb mismatch: idx={len(idx)} emb={Z.shape[0]} -> trimming to {k}")
        idx = idx.iloc[:k].reset_index(drop=True)
        Z = Z[:k]

    mids = idx["master_index"].astype(int).to_numpy()
    row = np.arange(len(mids), dtype=int)
    map_df = pd.DataFrame({"master_index": mids, f"{name}_row": row})
    return Z.astype(np.float32, copy=False), map_df

# Target transform

def transform_y(y: np.ndarray, mode: str):
    y = y.astype(float)
    offset = 0.0
    if mode == "none":
        return y, offset

    if mode in ("log1p", "log10p1"):
        miny = float(np.nanmin(y))
        # ensure (y + offset) >= 0 for log1p/log10p1
        if miny < 0.0:
            offset = -miny + 1e-6
        yp = y + offset
        yp = np.clip(yp, a_min=0.0, a_max=None)

        if mode == "log1p":
            return np.log1p(yp), offset
        else:
            return np.log10(1.0 + yp), offset

    raise ValueError(f"Unknown transform mode: {mode}")


def invert_y(y_t: np.ndarray, mode: str, offset: float):
    y_t = y_t.astype(float)
    if mode == "none":
        return y_t

    if mode == "log1p":
        return np.expm1(y_t) - offset
    if mode == "log10p1":
        return (10.0 ** y_t - 1.0) - offset

    raise ValueError(f"Unknown transform mode: {mode}")

# MLP model (PyTorch)

class MLP(nn.Module):
    def __init__(self, d_in: int, hidden_dims=(256, 128), dropout: float = 0.0):
        super().__init__()
        layers = []
        prev = d_in
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(
    X_train, y_train, X_val, y_val,
    hidden_dims=(256, 128),
    dropout=0.0,
    epochs=80,
    batch=256,
    lr=1e-3,
    weight_decay=1e-5,
    patience=10,
    device=torch.device("cpu"),
):
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch,
        shuffle=True,
        drop_last=False,
    )

    model = MLP(X_train.shape[1], hidden_dims=hidden_dims, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = float(loss_fn(val_pred, y_val_t).item())

        if val_loss < best_val - 1e-10:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if ep == 1 or ep % 10 == 0 or ep == epochs:
            print(f"Epoch {ep:03d}/{epochs}  val_MSE={val_loss:.6f}  best={best_val:.6f}  bad={bad}/{patience}")

        if bad >= patience:
            print("[info] early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        yhat_val = model(torch.tensor(X_val, dtype=torch.float32, device=device)).detach().cpu().numpy()

    return model, yhat_val

# CLI / main

def pick_merged(cli_path: str | None) -> Path:
    if cli_path:
        p = Path(cli_path)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"Merged PKL not found: {p}")
        return p

    candidates = sorted(MERGED_DIR.glob("merged_master_tabular_*.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No merged_master_tabular_*.pkl found in: {MERGED_DIR}")
    for p in candidates:
        if p.name.endswith("_0p5.pkl"):
            return p
    return candidates[0]

def auto_pick_target(df: pd.DataFrame) -> str:
    # Prefer columns that look like integrated/total flux
    cands = [c for c in df.columns if any(k in c.lower() for k in ["total", "int", "flux", "fint", "s_int"])]
    bad = ["opt", "ra", "dec", "err", "sigma"]
    cands = [c for c in cands if not any(b in c.lower() for b in bad)]
    for c in cands:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c

    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [c for c in numeric if c.lower() not in ["optra", "optdec", "ra", "dec", "master_index"]]
    if not numeric:
        raise ValueError("Could not auto-pick a numeric target column.")
    return numeric[0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--merged", type=str, default=None)
    p.add_argument("--target", type=str, default=None)
    p.add_argument("--use", choices=["img", "tab", "both"], default="both")
    p.add_argument("--transform", choices=["none", "log1p", "log10p1"], default="log1p")
    p.add_argument("--match-tol-arcsec", type=float, default=0.5)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--ridge-alpha", type=float, default=1.0)

    p.add_argument("--mlp-hidden", type=str, default="256,128")
    p.add_argument("--mlp-dropout", type=float, default=0.0)
    p.add_argument("--mlp-epochs", type=int, default=80)
    p.add_argument("--mlp-batch", type=int, default=256)
    p.add_argument("--mlp-lr", type=float, default=1e-3)
    p.add_argument("--mlp-weight-decay", type=float, default=1e-5)
    p.add_argument("--mlp-patience", type=int, default=10)

    p.add_argument("--device", type=str, default=None)

    # save paths
    p.add_argument("--ridge-out", type=str, default=str(DATA_DIR / "ridge_model.joblib"))
    p.add_argument("--mlp-out", type=str, default=str(DATA_DIR / "mlp_model.pt"))
    p.add_argument("--scaler-out", type=str, default=str(DATA_DIR / "feature_scaler.joblib"))

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    print(f"Device: {device.type}")

    # Load embeddings and index maps
    img_Z, img_map = None, None
    tab_Z, tab_map = None, None

    if args.use in ("img", "both"):
        img_Z, img_map = load_embeddings_with_index(IMG_EMB, IMG_IDX, "img")
        print(f"[info] img embeddings: {img_Z.shape}")

    if args.use in ("tab", "both"):
        tab_Z, tab_map = load_embeddings_with_index(TAB_EMB, TAB_IDX, "tab")
        print(f"[info] tab embeddings: {tab_Z.shape}")

    # Load merged target table
    merged_pkl = pick_merged(args.merged)
    print(f"[info] target source: {merged_pkl}")
    df = pd.read_pickle(merged_pkl)

    # drop dup columns
    df = df.loc[:, ~df.columns.duplicated()].copy()

    df = ensure_master_index(df, MASTER_CSV, match_tol_arcsec=float(args.match_tol_arcsec))

    target = args.target or auto_pick_target(df)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")
    print(f"[info] target column: {target}")

    y = pd.to_numeric(df[target], errors="coerce")
    tgt = pd.DataFrame({"master_index": df["master_index"].astype(int).to_numpy(), "y": y.to_numpy()})
    tgt = tgt.dropna(subset=["y"]).copy()

    # Intersection by master_index
    key = tgt
    if img_map is not None:
        key = key.merge(img_map, on="master_index", how="inner")
    if tab_map is not None:
        key = key.merge(tab_map, on="master_index", how="inner")

    print(f"[info] rows after intersection: {len(key)}")
    if len(key) < 200:
        raise RuntimeError("Too few matched rows to train. Check embeddings and target column.")

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

    # Train/test split
    indices = np.arange(len(y_t), dtype=int)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y_t, indices, test_size=args.test_size, random_state=args.seed
    )

    # Standardize X
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Ridge
    ridge = Ridge(alpha=args.ridge_alpha, random_state=args.seed)
    ridge.fit(X_train_s, y_train)
    y_pred_r_t = ridge.predict(X_test_s).astype(float)

    # MLP (with validation split from TRAIN)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_s, y_train, test_size=0.2, random_state=args.seed
    )

    hidden_dims = tuple(int(x) for x in args.mlp_hidden.split(",") if x.strip())

    mlp, _ = train_mlp(
        X_tr, y_tr, X_val, y_val,
        hidden_dims=hidden_dims,
        dropout=float(args.mlp_dropout),
        epochs=int(args.mlp_epochs),
        batch=int(args.mlp_batch),
        lr=float(args.mlp_lr),
        weight_decay=float(args.mlp_weight_decay),
        patience=int(args.mlp_patience),
        device=device,
    )

    mlp.eval()
    with torch.no_grad():
        y_pred_m_t = mlp(torch.tensor(X_test_s, dtype=torch.float32, device=device)).detach().cpu().numpy().astype(float)

    # Invert transform to raw flux space
    y_test_raw = invert_y(y_test, args.transform, y_offset)
    y_pred_r_raw = invert_y(y_pred_r_t, args.transform, y_offset)
    y_pred_m_raw = invert_y(y_pred_m_t, args.transform, y_offset)

    # Metrics computed in RAW space (the thing you care about)
    metrics = {
        "target": target,
        "use": args.use,
        "transform": args.transform,
        "y_offset": float(y_offset),
        "n_rows": int(len(key)),
        "feature_dim": int(X.shape[1]),
        "ridge": {
            "alpha": float(args.ridge_alpha),
            "mae": float(mean_absolute_error(y_test_raw, y_pred_r_raw)),
            "r2": float(r2_score(y_test_raw, y_pred_r_raw)),
        },
        "mlp": {
            "hidden": list(hidden_dims),
            "dropout": float(args.mlp_dropout),
            "epochs": int(args.mlp_epochs),
            "batch": int(args.mlp_batch),
            "lr": float(args.mlp_lr),
            "weight_decay": float(args.mlp_weight_decay),
            "patience": int(args.mlp_patience),
            "mae": float(mean_absolute_error(y_test_raw, y_pred_m_raw)),
            "r2": float(r2_score(y_test_raw, y_pred_m_raw)),
        },
    }

    # Predictions CSV (include both transformed and raw so plotting is easy)
    out_pred = pd.DataFrame({
        "master_index": key.iloc[idx_test]["master_index"].to_numpy(dtype=int),

        "y_true_raw": y_test_raw,
        "y_pred_ridge_raw": y_pred_r_raw,
        "y_pred_mlp_raw": y_pred_m_raw,

        "y_true_trans": y_test.astype(float),
        "y_pred_ridge_trans": y_pred_r_t,
        "y_pred_mlp_trans": y_pred_m_t,
    })

    out_metrics = DATA_DIR / "flux_model_metrics.json"
    out_preds = DATA_DIR / "flux_model_predictions.csv"
    out_metrics.write_text(json.dumps(metrics, indent=2))
    out_pred.to_csv(out_preds, index=False)

    print("[info] Saved:", out_metrics)
    print("[info] Saved:", out_preds)

    # SAVE MODELS (for demo)
    ridge_out = Path(args.ridge_out)
    mlp_out = Path(args.mlp_out)
    scaler_out = Path(args.scaler_out)

    ridge_out.parent.mkdir(parents=True, exist_ok=True)
    mlp_out.parent.mkdir(parents=True, exist_ok=True)
    scaler_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(ridge, ridge_out)
    joblib.dump(scaler, scaler_out)

    torch.save(
        {
            "state_dict": mlp.state_dict(),
            "in_dim": int(X.shape[1]),
            "hidden_dims": list(hidden_dims),
            "dropout": float(args.mlp_dropout),
            "target": target,
            "use": args.use,
            "transform": args.transform,
            "y_offset": float(y_offset),
        },
        mlp_out,
    )

    print("[info] Saved:", ridge_out)
    print("[info] Saved:", scaler_out)
    print("[info] Saved:", mlp_out)

    print("Ridge  MAE/R2:", metrics["ridge"]["mae"], metrics["ridge"]["r2"])
    print("MLP    MAE/R2:", metrics["mlp"]["mae"], metrics["mlp"]["r2"])
    print("DONE.")


if __name__ == "__main__":
    main()
