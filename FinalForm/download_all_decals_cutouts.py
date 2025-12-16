import argparse
import time
from pathlib import Path
import urllib.request
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd
import multiprocessing as mp

# Defaults for your project structure
ROOT = Path(__file__).resolve().parent
MASTER_CSV_DEFAULT = ROOT / "Data" / "gz_decals_gz2_master.csv"
IDS_PKL_DEFAULT    = ROOT / "Data" / "unique_master_indices.pkl"
OUTDIR_DEFAULT     = ROOT / "Images"

# Cutout parameters
DEFAULT_PIXSCALE = 0.262  # arcsec/pixel
DEFAULT_WIDTH    = 256    # pixels
DEFAULT_HEIGHT   = 256    # pixels

# Legacy Survey JPEG cutout base URL
BASE_URL = "https://www.legacysurvey.org/viewer/jpeg-cutout"


def build_url(ra_deg, dec_deg, pixscale, width, height):
    return (
        f"{BASE_URL}"
        f"?layer=dr8&bands=grz"
        f"&ra={ra_deg:.6f}&dec={dec_deg:.6f}"
        f"&pixscale={pixscale:.3f}"
        f"&width={width:d}&height={height:d}"
    )


def worker_download(task):
    (
        mid, ra, dec, outdir, pixscale, width, height,
        base_sleep, max_retries
    ) = task

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{mid:06d}.jpg"

    # Skip if already exists (resume-friendly)
    if out_path.exists():
        # tiny sleep so we don't spam even when skipping
        time.sleep(base_sleep * 0.1)
        return (mid, False, True)

    url = build_url(ra, dec, pixscale, width, height)
    backoff = base_sleep

    for attempt in range(max_retries):
        try:
            urllib.request.urlretrieve(url, out_path)
            # polite pause after a successful request
            time.sleep(base_sleep)
            return (mid, True, False)

        except HTTPError as e:
            if e.code == 429:
                # Too Many Requests – exponential backoff
                wait = backoff
                # Note: printing from workers is okay but can be noisy
                print(
                    f"[429] mid={mid} RA={ra:.3f} Dec={dec:.3f} "
                    f"attempt {attempt+1}/{max_retries}, sleeping {wait:.1f}s",
                    flush=True,
                )
                time.sleep(wait)
                backoff *= 2.0
                continue
            else:
                print(f"[error] HTTPError {e.code} for mid={mid}: {e}", flush=True)
                break

        except URLError as e:
            print(f"[error] URLError for mid={mid}: {e}", flush=True)
            # retry with backoff
            time.sleep(backoff)
            backoff *= 2.0
            continue

        except Exception as e:
            print(f"[error] Unexpected error for mid={mid}: {e}", flush=True)
            break

    # If we got here, all retries failed
    return (mid, False, False)


def parse_args():
    p = argparse.ArgumentParser(
        description="Download DECaLS JPEG cutouts for all used master_index IDs "
                    "with multiprocessing + rate limiting."
    )
    p.add_argument("--master-csv", type=str, default=str(MASTER_CSV_DEFAULT),
                   help="Path to gz_decals_gz2_master.csv (with ra/dec/master_index).")
    p.add_argument("--ids-pkl", type=str, default=str(IDS_PKL_DEFAULT),
                   help="Pickle with array of master_index values (e.g. unique_master_indices.pkl).")
    p.add_argument("--outdir", type=str, default=str(OUTDIR_DEFAULT),
                   help="Output directory for JPEG cutouts (default: Images/).")
    p.add_argument("--pixscale", type=float, default=DEFAULT_PIXSCALE,
                   help="Pixel scale in arcsec/pixel (default 0.262).")
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH,
                   help="Cutout width in pixels (default 256).")
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT,
                   help="Cutout height in pixels (default 256).")
    p.add_argument("--processes", type=int, default=3,
                   help="Number of parallel download processes (default 3).")
    p.add_argument("--sleep", type=float, default=0.3,
                   help="Base sleep (seconds) between successful requests per worker.")
    p.add_argument("--retries", type=int, default=5,
                   help="Max retries per object on 429/URLError.")
    return p.parse_args()


def main():
    args = parse_args()

    master_csv = Path(args.master_csv)
    ids_pkl    = Path(args.ids_pkl)
    outdir     = Path(args.outdir)

    print(f"[info] master CSV : {master_csv}")
    print(f"[info] IDs pickle : {ids_pkl}")
    print(f"[info] output dir : {outdir}")
    print(f"[info] pixscale={args.pixscale}, size={args.width}x{args.height}")
    print(f"[info] processes={args.processes}, base sleep={args.sleep}s, retries={args.retries}")

    # Load master catalog
    df = pd.read_csv(master_csv)
    if "master_index" not in df.columns:
        raise ValueError(f"'master_index' column not found in {master_csv}")
    if "ra" not in df.columns or "dec" not in df.columns:
        raise ValueError(f"'ra'/'dec' columns not found in {master_csv}")

    # Load list of master_index to download
    mid_list = pd.read_pickle(ids_pkl)
    mid_list = np.asarray(mid_list, dtype=int)

    # Subset master catalog to those indices
    df_sub = df.set_index("master_index").loc[mid_list]
    df_sub = df_sub.reset_index()
    total = len(df_sub)
    print(f"[info] will process {total} IDs (skipping any that already have files)")

    # Build tasks
    tasks = [
        (
            int(row["master_index"]),
            float(row["ra"]),
            float(row["dec"]),
            str(outdir),
            args.pixscale,
            args.width,
            args.height,
            args.sleep,
            args.retries,
        )
        for _, row in df_sub.iterrows()
    ]

    # Shared counters live in the main process; updated via callback
    processed = mp.Value("i", 0)
    downloaded_new = mp.Value("i", 0)
    skipped_existing = mp.Value("i", 0)
    t0 = time.time()

    def callback(result):
        """Called in main process when a worker finishes one task."""
        mid, new, skipped = result

        with processed.get_lock():
            processed.value += 1
            p_val = processed.value

        if new:
            with downloaded_new.get_lock():
                downloaded_new.value += 1
        if skipped:
            with skipped_existing.get_lock():
                skipped_existing.value += 1

        # Occasionally print progress + ETA
        if (p_val % 100 == 0) or (p_val == total):
            elapsed = time.time() - t0
            rate = p_val / max(elapsed, 1e-6)  # IDs per second
            pct = 100.0 * p_val / total
            remaining = total - p_val
            eta_sec = remaining / max(rate, 1e-6)
            eta_min = eta_sec / 60.0
            print(
                f"[progress] {p_val}/{total} ({pct:.2f}%) "
                f"new={downloaded_new.value}, existing={skipped_existing.value}, "
                f"rate={rate:.2f} IDs/s, ETA ~ {eta_min:.1f} min",
                flush=True,
            )

    print("[info] starting multiprocessing pool ...")
    with mp.Pool(processes=args.processes) as pool:
        for t in tasks:
            pool.apply_async(worker_download, args=(t,), callback=callback)

        pool.close()
        pool.join()

    print("[done] download phase complete.")
    print(f"       total processed  : {processed.value}")
    print(f"       newly downloaded : {downloaded_new.value}")
    print(f"       already existed  : {skipped_existing.value}")


if __name__ == "__main__":
    main()
