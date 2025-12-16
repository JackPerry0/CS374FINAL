from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
MERGED_DIR = ROOT / "Data" / "MergedDataset"

def main():
    pkl_files = sorted(MERGED_DIR.glob("merged_master_tabular_*.pkl"))
    if not pkl_files:
        print("No merged_master_tabular_*.pkl files found.")
        return

    all_ids = []

    for p in pkl_files:
        print(f"Loading {p.name} ...")
        df = pd.read_pickle(p)
        if "master_index" not in df.columns:
            raise ValueError(f"{p.name} has no 'master_index' column")
        all_ids.append(df["master_index"].values)

    all_ids = np.concatenate(all_ids)
    unique_ids = np.unique(all_ids).astype(int)

    print(f"\nTotal rows across all files: {len(all_ids)}")
    print(f"Unique galaxies (master_index): {len(unique_ids)}")

    out_path = ROOT / "Data" / "unique_master_indices.pkl"
    pd.to_pickle(unique_ids, out_path)
    print(f"Saved unique master_index array to: {out_path}")

if __name__ == "__main__":
    main()
