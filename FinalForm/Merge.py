import h5py 
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
from pathlib import Path

DATA_DIR = Path("Data")

MASTER_CSV_PATH = DATA_DIR / "gz_decals_gz2_master.csv"
TABULAR_PATH    = DATA_DIR / "lofar_opt_mass.pkl"

OUTPUT_PREFIX = str(Path("Data") / "MergedDataset" / "merged_master_tabular")
MATCH_RADII_ARCSEC = [0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]


def load_master_csv(path: Path) -> pd.DataFrame:

    print(f"Loading optical master CSV: {path}")
    df = pd.read_csv(path)

    # Expect columns "ra" and "dec" from DataGrab.py
    if "ra" not in df.columns or "dec" not in df.columns:
        raise ValueError(f"'ra'/'dec' not found in {path.name}. "
                         f"Columns are: {list(df.columns)}")

    # Rename to match the rest of the code (optRA/optDec)
    df = df.rename(columns={"ra": "optRA", "dec": "optDec"})

    print(f"  -> {len(df)} rows, columns: {list(df.columns)[:10]} ...")
    return df


def load_tabular(path: str) -> pd.DataFrame:
    print(f"Loading tabular pickle: {path}")
    df = pd.read_pickle(path)
    df = df.loc[:, ~df.columns.duplicated()]

    if "optRA" not in df.columns or "optDec" not in df.columns:
        raise ValueError("optRA/optDec not found in tabular dataset")

    print(f"  -> {len(df)} rows, {df.shape[1]} columns: {list(df.columns)[:10]} ...")
    return df


def spherical_to_xyz(ra_deg, dec_deg):
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.vstack((x, y, z)).T


def run_match(radius_arcsec, optical, tabular, xyz_opt, xyz_tab):

    radius_rad = np.deg2rad(radius_arcsec / 3600.0)
    chord_thresh = 2.0 * np.sin(radius_rad / 2.0)

    print(f"\n===== Matching with radius = {radius_arcsec} arcsec =====")

    tree = KDTree(xyz_tab)
    dist, idx = tree.query(xyz_opt, k=1)
    dist = dist.flatten()
    idx = idx.flatten()

    mask = dist <= chord_thresh
    matched_opt_idx = np.where(mask)[0]
    matched_tab_idx = idx[mask]

    print(f"Matches found: {len(matched_opt_idx)}")

    # Keep all columns from optical (including master_index, source, etc.)
    merged = optical.iloc[matched_opt_idx].reset_index(drop=True)
    merged = pd.concat(
        [merged, tabular.iloc[matched_tab_idx].reset_index(drop=True)],
        axis=1
    )

    print("first five merged ", merged[:5]["master_index"])
    
    #print("column heads")
    #for c in merged.columns:
    #    print(c)

    print("Merged shape:", merged.shape)

    # Output filename includes radius
    Path(OUTPUT_PREFIX).parent.mkdir(parents=True, exist_ok=True)
    out_path = f"{OUTPUT_PREFIX}_{str(radius_arcsec).replace('.','p')}.pkl"
    merged.to_pickle(out_path)
    print(f"Saved: {out_path}")

    return merged


def main():

    # Load data
    optical = load_master_csv(MASTER_CSV_PATH)
    tabular = load_tabular(TABULAR_PATH)

    # Drop NaNs in RA/Dec (required for KDTree)
    optical = optical.dropna(subset=["optRA", "optDec"])
    tabular = tabular.dropna(subset=["optRA", "optDec"])

    print(f"Optical entries (master): {len(optical)}")
    print(f"Tabular entries: {len(tabular)}")

    # Precompute xyz (expensive step, done once)
    xyz_opt = spherical_to_xyz(optical["optRA"].values, optical["optDec"].values)
    xyz_tab = spherical_to_xyz(tabular["optRA"].values, tabular["optDec"].values)

    # Try all radii
    for rad in MATCH_RADII_ARCSEC:
        run_match(rad, optical, tabular, xyz_opt, xyz_tab)


if __name__ == "__main__":
    main()
