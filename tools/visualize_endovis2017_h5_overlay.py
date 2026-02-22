"""Visualize EndoVis2017 train .h5 as overlay images.

Reads .h5 files under a directory. Each h5 is expected to contain:
  - image_left: (H, W, 3) uint8 (BGR order, as stored/used by cv2)
  - mask: (H, W) int or (1,H,W)
  - name: scalar bytes (optional)

Then colorize labels 1..7 using a fixed BGR colormap, blend with original
image, and save.

Usage:
  /data/yjh_files/miniconda3/envs/yjh/bin/python tools/visualize_endovis2017_h5_overlay.py \
    --input_dir /data/yjh_files/data/Endovis2017/data_h5/train \
    --out_dir /data/yjh_files/data/Endovis2017/vis_overlay_train \
    --alpha 0.4
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Iterable, Tuple

import cv2
import h5py
import numpy as np


# Label 1..7 colormap in **BGR** order (as requested)
CMAP_BGR = np.array(
    [
        (72, 203, 137),
        (39, 151, 187),
        (69, 179, 84),
        (151, 181, 50),
        (226, 185, 5),
        (191, 131, 137),
        (162, 109, 199),
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class H5Sample:
    image_bgr: np.ndarray  # (H,W,3) uint8
    mask: np.ndarray  # (H,W) int64
    name: str


def _safe_decode_name(x) -> str:
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8")
    return str(x)


def read_h5_sample(path: str, img_key_priority: Tuple[str, ...] = ("image_left", "image")) -> H5Sample:
    with h5py.File(path, "r") as f:
        img_key = None
        for k in img_key_priority:
            if k in f:
                img_key = k
                break
        if img_key is None:
            raise KeyError(f"No image key found in {path}. Tried: {img_key_priority}. Keys={list(f.keys())}")

        img = f[img_key][:]
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Unexpected image shape in {path}: {img.shape}")
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        mask = f["mask"][:]
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        if mask.ndim != 2:
            raise ValueError(f"Unexpected mask shape in {path}: {mask.shape}")
        mask = mask.astype(np.int64, copy=False)

        if "name" in f:
            name = _safe_decode_name(f["name"][()])
        else:
            name = os.path.splitext(os.path.basename(path))[0]

    return H5Sample(image_bgr=img, mask=mask, name=name)


def colorize_mask_bgr(mask: np.ndarray) -> np.ndarray:
    """Map labels 1..7 to BGR colors. 0 stays black."""
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for label in range(1, 8):
        out[mask == label] = CMAP_BGR[label - 1]
    return out

def _apply_color_overlay(bgr_img: np.ndarray, label_mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay a label mask on BGR image.

    label_mask: HxW, values in {0..7}, where 0=background
    """
    # BGR colors (OpenCV)
    # 1..7 correspond to EndoVis2017 categories order in README.
    colors = {
        1: (72,203,137),     
        2: (39, 151, 187),      
        3: (69, 179,84 ),     
        4: (151, 181,50 ),    
        5: (226, 185, 5),    
        6: (191, 131,137 ),    
        7: (162, 109,199 ), 
    }
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    out = rgb_img.copy()
    for cid, color in colors.items():
        m = (label_mask == cid)
        if not np.any(m):
            continue
        overlay = np.zeros_like(out, dtype=np.uint8)
        overlay[:, :] = color
        out[m] = (out[m] * (1 - alpha) + overlay[m] * alpha).astype(np.uint8)
    return out

def find_h5_files(input_dir: str) -> Iterable[str]:
    return sorted(glob.glob(os.path.join(input_dir, "*.h5")))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=False,default="/data/yjh_files/data/Endovis2017/data_h5/train")
    p.add_argument("--out_dir", type=str, required=False,default="/data/yjh_files/data/Endovis2017/vis_overlay_train")
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--limit", type=int, default=-1, help="Process only first N files (debug)")
    p.add_argument("--save_mask", action="store_true",default=True, help="Also save colorized mask only")
    p.add_argument("--ext", type=str, default=".png", choices=[".png", ".jpg", ".jpeg"])  # output format
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.save_mask:
        os.makedirs(os.path.join(args.out_dir, "mask"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "overlay"), exist_ok=True)

    files = list(find_h5_files(args.input_dir))
    if args.limit and args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"No .h5 files under {args.input_dir}")

    for i, path in enumerate(files):
        sample = read_h5_sample(path)
        # 使用你希望的 _apply_color_overlay 函数
        over = _apply_color_overlay(sample.image_bgr, sample.mask, alpha=float(args.alpha))

        stem = os.path.splitext(os.path.basename(path))[0]
        out_overlay = os.path.join(args.out_dir, "overlay", f"{stem}{args.ext}")
        # OpenCV 写文件按 BGR 保存即可
        cv2.imwrite(out_overlay, over)

        if args.save_mask:
            out_mask = os.path.join(args.out_dir, "mask", f"{stem}{args.ext}")
            color = colorize_mask_bgr(sample.mask)
            cv2.imwrite(out_mask, color)

        if (i + 1) % 200 == 0:
            print(f"[{i+1}/{len(files)}] saved: {out_overlay}")

    print(f"Done. Saved {len(files)} overlays to {os.path.join(args.out_dir, 'overlay')}")


if __name__ == "__main__":
    main()
