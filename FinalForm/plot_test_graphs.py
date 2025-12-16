import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TRUE_CANDIDATES = [
    "y_true", "y", "target", "flux", "flux_true", "true_flux",
    "radio_flux", "lofar_flux", "int_flux", "total_flux"
]

def pick_true_column(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    # exact candidate match first
    for c in TRUE_CANDIDATES:
        if c in cols:
            return c
    # fuzzy: any col containing these tokens
    tokens = ["true", "target", "y", "flux"]
    for c in cols:
        cl = c.lower()
        if any(t in cl for t in tokens) and ("pred" not in cl) and pd.api.types.is_numeric_dtype(df[c]):
            return c
    # fallback: first numeric col that doesn't look like a prediction
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    for c in num_cols:
        if "pred" not in c.lower():
            return c
    raise ValueError("Could not identify a true/target column in the predictions CSV.")

def pick_pred_columns(df: pd.DataFrame, true_col: str) -> list[str]:
    preds = []
    for c in df.columns:
        cl = c.lower()
        if c == true_col:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and (
            "pred" in cl or "ridge" in cl or "mlp" in cl or "yhat" in cl
        ):
            preds.append(c)
    # fallback: all numeric columns except true_col
    if not preds:
        preds = [c for c in df.columns if c != true_col and pd.api.types.is_numeric_dtype(df[c])]
    return preds

def mae(y, yhat):
    return float(np.mean(np.abs(y - yhat)))

def r2(y, yhat):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))

def parity_plot(y, yhat, title, outpath: Path):
    plt.figure()
    plt.scatter(y, yhat, s=6, alpha=0.6)
    lo = float(min(np.min(y), np.min(yhat)))
    hi = float(max(np.max(y), np.max(yhat)))
    plt.plot([lo, hi], [lo, hi])  # y=x line
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def residual_plot(y, yhat, title, outpath: Path):
    resid = yhat - y
    plt.figure()
    plt.scatter(y, resid, s=6, alpha=0.6)
    plt.axhline(0.0)
    plt.xlabel("True")
    plt.ylabel("Residual (Pred - True)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def residual_hist(y, yhat, title, outpath: Path):
    resid = yhat - y
    plt.figure()
    plt.hist(resid, bins=60)
    plt.xlabel("Residual (Pred - True)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", default="Data/flux_model_predictions.csv")
    ap.add_argument("--outdir", default="Data/test_graphs")
    args = ap.parse_args()

    pred_csv = Path(args.pred_csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_csv)
    true_col = pick_true_column(df)
    pred_cols = pick_pred_columns(df, true_col)

    y = df[true_col].to_numpy(dtype=float)

    print(f"[info] true column: {true_col}")
    print(f"[info] pred columns: {pred_cols}")

    for pc in pred_cols:
        yhat = df[pc].to_numpy(dtype=float)
        m = mae(y, yhat)
        r = r2(y, yhat)

        safe = pc.replace("/", "_")
        parity_plot(
            y, yhat,
            title=f"{pc}  (MAE={m:.3f}, R2={r:.3f})",
            outpath=outdir / f"parity_{safe}.png"
        )
        residual_plot(
            y, yhat,
            title=f"{pc} residuals vs true",
            outpath=outdir / f"residuals_{safe}.png"
        )
        residual_hist(
            y, yhat,
            title=f"{pc} residual histogram",
            outpath=outdir / f"resid_hist_{safe}.png"
        )

    print(f"[done] wrote plots to: {outdir.resolve()}")

if __name__ == "__main__":
    main()
