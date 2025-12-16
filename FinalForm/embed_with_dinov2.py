import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

import timm
from timm.data import resolve_model_data_config, create_transform


# Save every N images (adjust if you want more/less frequent checkpoints)
CHECKPOINT_INTERVAL = 10

# Paths relative to this script
ROOT = Path(__file__).resolve().parent
DEFAULT_IDS_PKL = ROOT / "Data" / "unique_master_indices.pkl"
DEFAULT_IMAGE_ROOT = ROOT / "Data" / "Images"
DEFAULT_OUT_PREFIX = ROOT / "Data" / "dinov2"


class GalaxyCutoutDataset(Dataset):
    def __init__(self, master_indices, image_root, transform):
        self.master_indices = list(map(int, master_indices))
        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self):
        return len(self.master_indices)

    def __getitem__(self, idx):
        mid = self.master_indices[idx]
        img_path = self.image_root / f"{mid:06d}.jpg"

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found for master_index={mid}: {img_path}")

        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        return x, mid


def parse_args():
    p = argparse.ArgumentParser(
        description="Embed galaxy images using DINOv2 (timm ViT-B/14)."
    )
    p.add_argument(
        "--ids-pkl",
        type=str,
        default=str(DEFAULT_IDS_PKL),
        help=f"Pickle with master_index values (default: {DEFAULT_IDS_PKL})",
    )
    p.add_argument(
        "--image-root",
        type=str,
        default=str(DEFAULT_IMAGE_ROOT),
        help=f"Directory with cutout images (default: {DEFAULT_IMAGE_ROOT})",
    )
    p.add_argument(
        "--out-prefix",
        type=str,
        default=str(DEFAULT_OUT_PREFIX),
        help=f"Output prefix (default: {DEFAULT_OUT_PREFIX})",
    )
    p.add_argument("--batch", type=int, default=64, help="Batch size.")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="torch device string: 'cuda', 'mps', or 'cpu'. If not set, auto-detect.",
    )
    return p.parse_args()


def choose_device(cli_device: str | None) -> torch.device:
    if cli_device:
        return torch.device(cli_device)
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon / Metal
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    args = parse_args()
    device = choose_device(args.device)
    print(f"[info] device={device.type}, batch={args.batch}")

    ids_path = Path(args.ids_pkl)
    image_root = Path(args.image_root)
    out_prefix = Path(args.out_prefix)

    if not ids_path.exists():
        raise FileNotFoundError(f"IDs pickle not found: {ids_path}")

    master_indices = pd.read_pickle(ids_path)
    master_indices = np.asarray(master_indices, dtype=int)
    print(f"[info] loaded {len(master_indices)} master_index values from {ids_path}")

    existing_ids = []
    missing_count = 0
    for mid in master_indices:
        img_path = image_root / f"{mid:06d}.jpg"
        if img_path.exists():
            existing_ids.append(mid)
        else:
            missing_count += 1

    existing_ids = np.asarray(existing_ids, dtype=int)
    print(f"[info] found images for {len(existing_ids)} IDs")
    print(f"[info] missing images for {missing_count} IDs (these will be skipped)")

    if len(existing_ids) == 0:
        raise RuntimeError(
            "No images found for any master_index; "
            "check --image-root and filename pattern."
        )

    model_name = "vit_base_patch14_dinov2.lvd142m"
    print(f"[info] loading model: {model_name}")
    model = timm.create_model(model_name, pretrained=True)
    # Remove classifier head -> pure feature extractor
    if hasattr(model, "reset_classifier"):
        model.reset_classifier(0)
    model.eval()
    model.to(device)

    # Get the recommended transforms for this model
    cfg = resolve_model_data_config(model)
    transform = create_transform(**cfg)

    feat_dim = model.num_features
    print(f"[info] DINOv2 feature dimension = {feat_dim}")

    ds = GalaxyCutoutDataset(existing_ids, image_root, transform)
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    N = len(ds)
    print(f"[info] embedding N={N} images (IDs with existing files)")

    embeddings = np.zeros((N, feat_dim), dtype=np.float32)
    out_ids = np.zeros((N,), dtype=np.int64)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    emb_path = Path(str(out_prefix) + "_embeddings.npy")
    idx_path = Path(str(out_prefix) + "_index.csv")

    # Helper: save a partial checkpoint
    def save_checkpoint(num_filled: int, elapsed_sec: float):
        if num_filled == 0:
            return

        # Save only the filled portion
        np.save(emb_path, embeddings[:num_filled])
        df_idx = pd.DataFrame(
            {
                "row_id": np.arange(num_filled, dtype=int),
                "master_index": out_ids[:num_filled],
            }
        )
        df_idx.to_csv(idx_path, index=False)

        pct = 100.0 * num_filled / N
        if elapsed_sec > 0:
            rate = num_filled / elapsed_sec  # images per second
            remaining = N - num_filled
            eta_sec = remaining / max(rate, 1e-8)
            eta_min = eta_sec / 60.0
        else:
            eta_min = float("inf")

        print(
            f"[checkpoint] saved {num_filled}/{N} "
            f"({pct:.2f}%) embeddings; ETA ~ {eta_min:.1f} min"
        )

    # Embedding loop with progress + ETA + checkpoints
    idx = 0
    last_checkpoint_count = 0
    t0 = time.time()

    model.eval()
    with torch.no_grad():
        for batch_x, batch_ids in dl:
            batch_x = batch_x.to(device)
            feats = model(batch_x)  # (B, feat_dim)
            feats = feats.cpu().numpy()

            B = feats.shape[0]
            embeddings[idx : idx + B, :] = feats
            out_ids[idx : idx + B] = batch_ids.numpy()
            idx += B

            # Time for a checkpoint?
            if (idx - last_checkpoint_count) >= CHECKPOINT_INTERVAL or idx == N:
                elapsed = time.time() - t0
                save_checkpoint(idx, elapsed)
                last_checkpoint_count = idx

    print(f"[done] finished embedding {N} images.")
    print(f"       final embeddings: {emb_path}")
    print(f"       final index CSV: {idx_path}")


if __name__ == "__main__":
    main()
