"""Export EndoVis2018 test GT binary masks from per-image H5 files.

Input:
  /data/yjh_files/data/Endovis2018/data_h5/test/*.h5

Each .h5 is expected to contain dataset:
  - "mask": HxW integer mask with values in [0..7]

Output:
  /data/yjh_files/indice/gt/<image_name>/<label>.png

Where:
  - <image_name> defaults to the h5 filename stem (e.g., seq_2_frame000)
  - <label> is one of "1".."7" (as requested)

Binary mask values:
  - foreground (mask == label): 255
  - background: 0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import cv2
from tqdm import tqdm


def _read_name_from_h5(f: h5py.File, fallback: str) -> str:
    """Read image name from h5 dataset `name` if possible."""
    if "name" not in f:
        return fallback
    name_ds = f["name"][...]
    # Could be bytes, numpy bytes_, or scalar array.
    try:
        if isinstance(name_ds, (bytes, np.bytes_)):
            return name_ds.decode("utf-8")
        if np.isscalar(name_ds):
            return str(name_ds)
        # 0-d or 1-d array
        if hasattr(name_ds, "shape") and name_ds.shape == ():
            v = name_ds.item()
            return v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v)
    except Exception:
        pass
    return fallback


def export_one(h5_path: Path, out_root: Path, labels: list[int]) -> None:
    stem = h5_path.stem
    with h5py.File(str(h5_path), "r") as f:
        if "mask" not in f:
            raise KeyError(f"{h5_path} missing dataset 'mask'. keys={list(f.keys())}")

        mask = f["mask"][...]
        if mask.ndim != 2:
            raise ValueError(f"{h5_path} mask must be 2D (HxW), got shape={mask.shape}")

        name = _read_name_from_h5(f, fallback=stem)

    # Ensure integer
    mask = np.asarray(mask)
    if not np.issubdtype(mask.dtype, np.integer):
        mask = mask.astype(np.int64)

    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    for lb in labels:
        if lb not in np.unique(mask):
            print(f"Label {lb} not found in mask {name}")
            continue
        bin_mask = (mask == lb).astype(np.uint8) * 255
        out_path = out_dir / f"{lb}.png"
        ok = cv2.imwrite(str(out_path), bin_mask)
        if not ok:
            raise IOError(f"cv2.imwrite failed: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--h5_dir",
        type=str,
        default="/data/yjh_files/data/Endovis2018/data_h5/test",
        help="Directory containing EndoVis2018 test h5 files",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="/data/yjh_files/indice/gt",
        help="Output directory root",
    )
    p.add_argument(
        "--labels",
        type=str,
        default="1,2,3,4,5,6,7",
        help="Comma-separated labels to export",
    )
    p.add_argument(
        "--glob",
        type=str,
        default="*.h5",
        help="Glob pattern within h5_dir",
    )
    args = p.parse_args()

    h5_dir = Path(args.h5_dir)
    out_dir = Path(args.out_dir)
    labels = [int(x) for x in args.labels.split(",") if x.strip()]

    if not h5_dir.exists():
        raise FileNotFoundError(f"h5_dir not found: {h5_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    h5_paths = sorted(h5_dir.glob(args.glob))
    if not h5_paths:
        raise FileNotFoundError(f"No h5 files found: {h5_dir}/{args.glob}")

    for hp in tqdm(h5_paths, desc="export_gt", unit="h5"):
        export_one(hp, out_dir, labels)

    print(f"[DONE] exported {len(h5_paths)} h5 into: {out_dir}")


if __name__ == "__main__":
    main()
