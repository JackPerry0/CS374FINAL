from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

# Data folder (FinalForm/Data/)
DATA_DIR = ROOT / "Data"

def load_catalog(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    print(f"Loading {path} ...")
    df = pd.read_csv(path)
    print(f"  -> {len(df)} rows, {len(df.columns)} columns")
    return df


def normalize_ra_dec(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {}
    # RA
    for cand in ["ra", "RA", "Ra"]:
        if cand in df.columns:
            colmap[cand] = "ra"
            break
    # Dec
    for cand in ["dec", "DEC", "Dec"]:
        if cand in df.columns:
            colmap[cand] = "dec"
            break

    df = df.rename(columns=colmap)

    if "ra" not in df.columns or "dec" not in df.columns:
        raise ValueError(
            "Could not find RA/Dec columns in dataframe. "
            f"Columns are: {list(df.columns)}"
        )

    return df

def main():
    # Load the three catalogs
    gz2 = load_catalog("gz2_hart16.csv")
    ab  = load_catalog("gz_decals_volunteers_ab.csv")
    c   = load_catalog("gz_decals_volunteers_c.csv")

    # Normalize RA/Dec names
    gz2 = normalize_ra_dec(gz2)
    ab  = normalize_ra_dec(ab)
    c   = normalize_ra_dec(c)

    # Tag each source so we know where rows came from
    gz2["source"] = "gz2"
    ab["source"]  = "decals_ab"
    c["source"]   = "decals_c"

    # Keep RA/Dec plus any obvious ID columns if present
    base_cols = ["ra", "dec", "source"]
    id_candidates = ["iauname", "dr8_objid", "iauname_dr8", "objid", "id"]

    def subset(df):
        cols = base_cols.copy()
        for col in id_candidates:
            if col in df.columns and col not in cols:
                cols.append(col)
        return df[cols]

    gz2_sub = subset(gz2)
    ab_sub  = subset(ab)
    c_sub   = subset(c)
    
    # Combine into one big table
    full = pd.concat([gz2_sub, ab_sub, c_sub], ignore_index=True)
    print(f"\nCombined rows (with duplicates): {len(full)}")
    
    print("example dec values", full[:5]["dec"])
    
    full = full.drop_duplicates(subset=["ra","dec"]).reset_index(drop=True)
    print(f"\nCombined rows (without exact duplicates): {len(full)}")

    # Optional: deduplicate by rounded RA/Dec ONE OF THE DATASETS IS MORE PRECISE THEN THE OTHER TWO
    full["ra_round"]  = full["ra"].round(5)
    full["dec_round"] = full["dec"].round(5)
    
    print("example rounded dec values", full[:5]["dec"])

    full_unique = full.drop_duplicates(subset=["ra_round", "dec_round"]).reset_index(drop=True)
    full_unique["master_index"] = full_unique.index

    print(f"Unique rows after removing rounded RA/Dec duplicate: {len(full_unique)}")

    print("example master index", full_unique[:5]["master_index"])

    # Save master table into Data/
    out_path = DATA_DIR / "gz_decals_gz2_master.csv"
    full_unique.to_csv(out_path, index=False)
    print(f"\nSaved master RA/Dec table to: {out_path}")

    #we have three identifiers: the original ra dec value, the rounded ra dec value, and master index
if __name__ == "__main__":
    main()
