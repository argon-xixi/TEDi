"""Overlay a query binary mask PNG on the corresponding EndoVis2018 h5 image.

Given a query PNG path like:
  /data/yjh_files/indice/baseline_2/seq_2_frame002/1_2_0.958..._0.959..._0.919....png

We will:
  1) Infer image_name from its parent folder name (e.g., seq_2_frame002)
  2) Find the same-name .h5 under /data/yjh_files/data/Endovis2018/data_h5/**/seq_2_frame002.h5
  3) Read the image (dataset: image_left or image)
  4) Parse label from query filename (2nd field split by '_', e.g., '2')
  5) Read query png as binary mask (foreground = pixel>0)
  6) Resize mask to image size if needed (nearest)
  7) Blend the label color onto the image and save.

Color map (BGR):
    colors = {
        1: (72,203,137),
        2: (39, 151, 187),
        3: (69, 179, 84),
        4: (151, 181, 50),
        5: (226, 185, 5),
        6: (191, 131, 137),
        7: (162, 109, 199),
    }

Notes about color/order:
  - EndoVis2018 h5 files in this repo are typically stored as RGB (see endovis2018_h5.py).
  - This script converts the h5 image to BGR before overlay so that cv2.imwrite saves correct colors.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, Tuple

import cv2
import h5py
import numpy as np


COLORS_BGR: Dict[int, Tuple[int, int, int]] = {
    # 1: (72, 203, 137),
    1: (226, 185, 5),
    2: (39, 151, 187),
    # 3: (191, 131, 137),
    3: (69, 179, 84),
    4: (151, 181, 50),
    5: (72, 203, 137),
    6: (191, 131, 137),
    7: (162, 109, 199),
}


def parse_label_from_query_png(query_png: Path) -> int:
    """Label is the second field in filename split by '_'"""
    # Example: 1_2_0.95_0.95_0.91.png -> label=2
    stem = query_png.stem
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unexpected query filename: {query_png.name}")
    try:
        return int(parts[1])
    except Exception as e:
        raise ValueError(f"Cannot parse label from {query_png.name}") from e


def infer_image_name(query_png: Path) -> str:
    # Parent directory name is expected to be image_name, e.g. seq_2_frame002
    if query_png.parent is None:
        raise ValueError(f"Cannot infer image_name from path: {query_png}")
    return query_png.parent.name


def find_h5_by_image_name(h5_root: Path, image_name: str) -> Path:
    # Search recursively
    pattern = str(h5_root / "**" / f"{image_name}.h5")
    matches = list(Path(h5_root).glob(f"**/{image_name}.h5"))
    if not matches:
        raise FileNotFoundError(f"Cannot find h5 for {image_name} under {h5_root} (pattern={pattern})")
    # If multiple, prefer test over train? pick shortest path
    matches = sorted(matches, key=lambda p: (len(str(p)), str(p)))
    return matches[0]


def read_image_from_h5(h5_path: Path) -> np.ndarray:
    """Return image as BGR uint8."""
    with h5py.File(str(h5_path), "r") as f:
        key = None
        for k in ("image_left", "image"):
            if k in f:
                key = k
                break
        if key is None:
            raise KeyError(f"No image dataset found in {h5_path}. keys={list(f.keys())}")
        img = f[key][...]

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Unexpected image shape in {h5_path}: {img.shape}")
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # EndoVis2018 h5 image is stored as RGB in this repo, convert to BGR for cv2
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img_bgr


def read_query_mask_binary(query_png: Path) -> np.ndarray:
    """Read query png and convert to binary mask (uint8 0/1)."""
    m = cv2.imread(str(query_png), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Cannot read query png: {query_png}")
    if m.ndim == 3:
        m = np.sum(m, axis=-1)  # convert to grayscale by summing channels
    m = np.asarray(m)
    return (m > 0).astype(np.uint8)


def overlay_binary_mask(
    image_bgr: np.ndarray,
    mask01: np.ndarray,
    label: int,
    alpha: float = 0.6,
    colors_bgr: Dict[int, Tuple[int, int, int]] = COLORS_BGR,
) -> np.ndarray:
    """Overlay a binary mask onto a BGR image with a label color."""
    if label not in colors_bgr:
        raise KeyError(f"Unknown label {label}. Available: {sorted(colors_bgr.keys())}")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"image_bgr must be HxWx3, got {image_bgr.shape}")
    if mask01.ndim != 2:
        raise ValueError(f"mask01 must be HxW, got {mask01.shape}")

    h, w = image_bgr.shape[:2]
    if mask01.shape != (h, w):
        mask01 = cv2.resize(mask01, (w, h), interpolation=cv2.INTER_NEAREST)

    m = mask01.astype(bool)
    if not np.any(m):
        return image_bgr.copy()

    out = image_bgr.copy()
    color = np.array(colors_bgr[label], dtype=np.float32)
    # Blend only masked pixels
    out_f = out.astype(np.float32)
    out_f[m] = out_f[m] * (1.0 - alpha) + color * alpha
    return np.clip(out_f, 0, 255).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--query_png",
        type=str,
        default="/data/yjh_files/indice/baseline_2/seq_2_frame079/2_6_0.9724248051643372_0.9115099906921387_0.8863749504089355.png",
        help="Query mask png path under baseline_2/.../<image_name>/<query>_<label>_*.png",
    )
    ap.add_argument(
        "--h5_root",
        type=str,
        default="/data/yjh_files/data/Endovis2018/data_h5",
        help="Root directory containing train/test h5 subfolders",
    )
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument(
        "--label",
        type=int,
        default=-1,
        help="Override label (default: parse from filename second field)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="Output overlay image path (default: alongside query png)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    query_png = Path(args.query_png)
    if not query_png.exists():
        raise FileNotFoundError(query_png)

    image_name = infer_image_name(query_png)
    label = args.label if args.label > 0 else parse_label_from_query_png(query_png)

    h5_path = find_h5_by_image_name(Path(args.h5_root), image_name)
    img_bgr = read_image_from_h5(h5_path)
    mask01 = read_query_mask_binary(query_png)
    over_bgr = overlay_binary_mask(img_bgr, mask01, label=label, alpha=float(args.alpha))

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = query_png.with_name(query_png.stem + f"_overlay_label{label}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), over_bgr)
    if not ok:
        raise IOError(f"cv2.imwrite failed: {out_path}")

    print("[DONE] overlay saved:", out_path)
    print("        image_name:", image_name)
    print("        h5_path:   ", h5_path)
    print("        label:     ", label)


if __name__ == "__main__":
    main()
