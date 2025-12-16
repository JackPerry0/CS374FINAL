import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TRUE_CANDIDATES = [
    "y_true", "true", "target", "y", "flux_true", "true_flux", "flux",
    "lofar_flux", "int_flux", "total_flux"
]

def pick_true_column(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    for c in TRUE_CANDIDATES:
        if c in cols and pd.api.types.is_numeric_dtype(df[c]):
            return c
    # fallback: first numeric column that doesn't look like a pred
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]) and "pred" not in c.lower():
            return c
    raise ValueError("Could not find true/target column in predictions CSV.")

def pick_pred_columns(df: pd.DataFrame, true_col: str) -> list[str]:
    preds = []
    for c in df.columns:
        if c == true_col:
            continue
        cl = c.lower()
        if pd.api.types.is_numeric_dtype(df[c]) and ("pred" in cl or "ridge" in cl or "mlp" in cl or "yhat" in cl):
            preds.append(c)
    if not preds:
        preds = [c for c in df.columns if c != true_col and pd.api.types.is_numeric_dtype(df[c])]
    return preds

def keep_only_raw_pred_cols(pred_cols: list[str]) -> list[str]:
    """
    Keep only columns that look like RAW predictions.
    Common patterns: *_raw, *raw*, etc.
    Exclude anything that looks transformed.
    """
    raw = []
    for c in pred_cols:
        cl = c.lower()
        if "raw" in cl and "trans" not in cl:
            raw.append(c)
    return raw

def transform(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return y
    if mode == "log1p":
        return np.log1p(np.clip(y, a_min=0, a_max=None))
    if mode == "log10p1":
        return np.log10(1.0 + np.clip(y, a_min=0, a_max=None))
    raise ValueError(f"Unknown transform mode: {mode}")

def corrcoef(a, b) -> float:
    a = np.asarray(a); b = np.asarray(b)
    if a.size < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

def mae(a, b) -> float:
    return float(np.mean(np.abs(a - b)))

def medae(a, b) -> float:
    return float(np.median(np.abs(a - b)))

def std_resid(a, b) -> float:
    return float(np.std(b - a))

def qclip_limits(x: np.ndarray, q: float):
    lo = float(np.quantile(x, q))
    hi = float(np.quantile(x, 1 - q))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.min(x)), float(np.max(x))
    return lo, hi

def reportcard_2x2(y_true, y_pred, model_name: str, title_prefix: str,
                   outpath: Path, clip_q: float = 0.01, max_points: int = 8000,
                   seed: int = 42):
    rng = np.random.default_rng(seed)

    # Downsample for scatter plots only
    n = len(y_true)
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        yt_sc = y_true[idx]
        yp_sc = y_pred[idx]
    else:
        yt_sc = y_true
        yp_sc = y_pred

    resid = y_pred - y_true
    resid_sc = yp_sc - yt_sc

    # Metrics (computed on full arrays)
    c = corrcoef(y_true, y_pred)
    m = mae(y_true, y_pred)
    md = medae(y_true, y_pred)
    s = std_resid(y_true, y_pred)

    # Robust plot limits to avoid outliers nuking readability
    xlo, xhi = qclip_limits(y_true, clip_q)
    ylo, yhi = qclip_limits(y_pred, clip_q)
    rlo, rhi = qclip_limits(resid, clip_q)

    # Bins for distribution overlay
    combined = np.concatenate([y_true, y_pred])
    bins = np.histogram_bin_edges(combined[np.isfinite(combined)], bins=50)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5))
    fig.suptitle(f"{title_prefix} - {model_name}\nCorr={c:.4f}, MAE={m:.3f}, MedAE={md:.3f}, Std={s:.3f}")

    # (1) True vs Predicted
    ax = axes[0, 0]
    ax.scatter(yt_sc, yp_sc, s=6, alpha=0.6)
    ax.set_title("True vs Predicted")
    ax.set_xlabel("True Flux")
    ax.set_ylabel("Predicted Flux")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)

    # (2) Residual Distribution
    ax = axes[0, 1]
    ax.hist(resid, bins=50, edgecolor="black", alpha=0.8)
    ax.set_title("Residual Distribution")
    ax.set_xlabel("Residual = Pred - True")
    ax.set_ylabel("Count")
    ax.set_xlim(rlo, rhi)

    # (3) Residuals vs Predicted
    ax = axes[1, 0]
    ax.scatter(yp_sc, resid_sc, s=6, alpha=0.6)
    ax.axhline(0.0, linestyle="--")
    ax.set_title("Residuals vs Predicted")
    ax.set_xlabel("Predicted Flux")
    ax.set_ylabel("Residual")
    ax.set_ylim(rlo, rhi)

    # (4) Flux Distribution Comparison
    ax = axes[1, 1]
    ax.hist(y_true, bins=bins, alpha=0.6, label="True")
    ax.hist(y_pred, bins=bins, alpha=0.6, label="Predicted")
    ax.set_title("Flux Distribution Comparison")
    ax.set_xlabel("Flux")
    ax.set_ylabel("Count")
    ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", default="Data/flux_model_predictions.csv")
    ap.add_argument("--outdir", default="Data/test_graphs")
    ap.add_argument("--transform", choices=["none", "log1p", "log10p1"], default="log1p",
                    help="Apply transform to true and predicted before plotting/metrics.")
    ap.add_argument("--title_prefix", default="RAW (log flux)")
    ap.add_argument("--clip_q", type=float, default=0.01, help="Quantile clip for plot limits (0.01 = 1%%/99%%).")
    ap.add_argument("--max_points", type=int, default=8000, help="Max points for scatter plots.")
    args = ap.parse_args()

    pred_csv = Path(args.pred_csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_csv)
    true_col = pick_true_column(df)
    pred_cols = pick_pred_columns(df, true_col)

    # ONLY keep RAW prediction columns
    pred_cols = keep_only_raw_pred_cols(pred_cols)
    if not pred_cols:
        raise ValueError(
            "No RAW prediction columns found. Expected prediction columns to include 'raw' "
            "(e.g., *_raw). If your CSV uses different naming, adjust the filter."
        )

    y_true_raw = df[true_col].to_numpy(dtype=float)

    for pc in pred_cols:
        y_pred_raw = df[pc].to_numpy(dtype=float)

        y_true = transform(y_true_raw, args.transform)
        y_pred = transform(y_pred_raw, args.transform)

        safe = pc.replace("/", "_")
        outpath = outdir / f"reportcard_{safe}.png"

        reportcard_2x2(
            y_true=y_true,
            y_pred=y_pred,
            model_name=pc,
            title_prefix=args.title_prefix,
            outpath=outpath,
            clip_q=args.clip_q,
            max_points=args.max_points
        )
        print(f"[ok] wrote {outpath}")

    print(f"[done] all plots in {outdir.resolve()}")

if __name__ == "__main__":
    main()
