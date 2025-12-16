from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import h5py

PKL_NAMES = ["merged_optical_tabular_0p1.pkl", "merged_optical_tabular_0p2.pkl"]
H5_NAME = "Galaxy10_DECals_sample.h5"

COLS = [
    # reembed_tabular.py (full + enriched)
    "mag_g","mag_r","mag_z","mag_w1","mag_w2",
    "magerr_g","magerr_r","magerr_z","magerr_w1","magerr_w2",
    "optRA","optDec",
    "z_best","zphot","zphot_err","flag_qual",
    "g_rest","r_rest","z_rest","U_rest","V_rest","J_rest","K_rest","w1_rest","w2_rest",
    "Mass_median","Mass_l68","Mass_u68",
    "r_50","r_50_err",
    "pstar",
    # vectorize_data.py expects these extra tabular cols
    "flag_mass","Legacy_Coverage",
    # target used by train_flux_models.py
    "Total_flux",
]

def main(n: int = 200, seed: int = 123) -> None:
    root = Path(__file__).resolve().parent
    rng = np.random.default_rng(seed)

    df = pd.DataFrame(index=np.arange(n, dtype=int))

    for c in ["mag_g","mag_r","mag_z","mag_w1","mag_w2"]:
        df[c] = rng.uniform(14, 26, size=n)

    for c in ["magerr_g","magerr_r","magerr_z","magerr_w1","magerr_w2"]:
        df[c] = rng.uniform(0.001, 0.2, size=n)

    df["optRA"] = rng.uniform(0, 360, size=n)
    df["optDec"] = rng.uniform(-30, 30, size=n)

    df["z_best"] = rng.uniform(0, 1.0, size=n)
    df["zphot"] = df["z_best"] + rng.normal(0, 0.05, size=n)
    df["zphot_err"] = rng.uniform(0.001, 0.2, size=n)

    df["flag_qual"] = rng.integers(0, 5, size=n)
    df["flag_mass"] = rng.integers(0, 2, size=n)
    df["Legacy_Coverage"] = rng.uniform(0, 1, size=n)
    df["pstar"] = rng.uniform(0, 1, size=n)

    for c in ["g_rest","r_rest","z_rest","U_rest","V_rest","J_rest","K_rest","w1_rest","w2_rest"]:
        df[c] = rng.uniform(-25, -10, size=n)

    df["Mass_median"] = rng.uniform(8, 12, size=n)      # log10(M/Msun)-ish
    df["Mass_l68"] = df["Mass_median"] - rng.uniform(0.0, 0.5, size=n)
    df["Mass_u68"] = df["Mass_median"] + rng.uniform(0.0, 0.5, size=n)

    df["r_50"] = rng.uniform(0.1, 10.0, size=n)
    df["r_50_err"] = rng.uniform(0.0, 0.5, size=n)

    # Target flux: strictly positive (log-normal)
    df["Total_flux"] = np.exp(rng.normal(0.0, 1.0, size=n)).astype(float)

    # Ensure all expected columns exist
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Bug in generator: missing columns {missing}")

    # Save PKLs
    for name in PKL_NAMES:
        out = root / name
        df.to_pickle(out)
        print("[ok] wrote", out)

    H, W = 64, 64
    imgs = rng.integers(0, 256, size=(n, H, W, 3), dtype=np.uint8)

    h5_path = root / H5_NAME
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("images", data=imgs, compression="gzip", compression_opts=4)
    print("[ok] wrote", h5_path)

    print("\nNext steps (from this folder):")
    print("  1) python vectorize_data.py")
    print("  2) python reembed_tabular.py")
    print("  3) python train_flux_models.py")

if __name__ == "__main__":
    main()
