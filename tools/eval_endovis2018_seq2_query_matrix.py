"""Evaluate query correctness for EndoVis2018 seq_2 using GT binary masks.

You described the following folder structure:

baseline_dir (/data/yjh_files/indice/baseline_1):
  <image_name>/                     # 一级标题：图片名，如 seq_2_frame000
    <query>_<label>_<cls>_<mask>_*.png   # 二级标题：query png

Where:
  - query: int in [0..9]
  - label: int (1..7)
  - cls: class confidence (float)
  - mask: mask confidence (float)

Task:
  For seq_2 images only, for each image and each query (0..9):
    - If no prediction file for this query satisfies:
        cls_conf > 0.85 and mask_conf > 0.8
      -> matrix[q, img_idx] = 0
    - Otherwise, for each satisfying prediction:
        - Find GT mask by (image_name, label) in gt_dir:
            gt_dir/<image_name>/<label>.png
          If not found -> treat as wrong.
        - Convert predicted png and GT png to binary foreground/background.
        - Resize both to 128x128.
        - Compute IoU. If any candidate IoU > 0.75 -> correct.
      -> matrix[q, img_idx] = 1 if correct else 2

Outputs:
  - .npy matrix (int64) of shape (10, N)
  - .csv for human inspection
  - optional .json debug stats
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


PRED_RE = re.compile(
    r"^(?P<q>\d+)_(?P<label>\d+)_(?P<cls>[0-9]*\.?[0-9]+)_(?P<mask>[0-9]*\.?[0-9]+)_(?P<rest>.+)\.png$"
)


@dataclass(frozen=True)
class PredItem:
    q: int
    label: int
    cls_conf: float
    mask_conf: float
    path: Path


def parse_pred_filename(p: Path) -> Optional[PredItem]:
    m = PRED_RE.match(p.name)
    if not m:
        return None
    try:
        return PredItem(
            q=int(m.group("q")),
            label=int(m.group("label")),
            cls_conf=float(m.group("cls")),
            mask_conf=float(m.group("mask")),
            path=p,
        )
    except Exception:
        return None


def read_mask_as_binary(path: Path, threshold: int = 0) -> np.ndarray:
    """Read an image as binary mask (uint8 0/1).

    Prediction png may not be binary; we treat any value > threshold as foreground.
    Works for grayscale or color images.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if img.ndim == 3:
        # BGR or BGRA -> take first channel
        img=np.sum(img, axis=-1)
    # Ensure numeric
    img = np.asarray(img)
    bin_mask = (img > threshold).astype(np.uint8)
    return bin_mask


def resize_bin(mask01: np.ndarray, size: int = 128) -> np.ndarray:
    # Nearest to avoid interpolation artifacts.
    return cv2.resize(mask01, (size, size), interpolation=cv2.INTER_NEAREST)


def iou(mask_a01: np.ndarray, mask_b01: np.ndarray) -> float:
    a = mask_a01.astype(bool)
    b = mask_b01.astype(bool)
    inter = np.logical_and(a, b).sum(dtype=np.int64)
    a_sum= a.sum(dtype=np.int64)
    return float(inter) / float(a_sum) if a_sum > 0 else 1.0
    # union = np.logical_or(a, b).sum(dtype=np.int64)
    # if union == 0:
    #     return 1.0
    # return float(inter) / float(union)


def list_seq2_images(baseline_dir: Path, seq_prefix: str = "seq_2_") -> List[Path]:
    # baseline_dir contains per-image directories
    imgs = [p for p in baseline_dir.iterdir() if p.is_dir() and p.name.startswith(seq_prefix)]
    # Sort by frame number if present
    def key_fn(p: Path):
        # seq_2_frame000 -> 0
        m = re.search(r"frame(\d+)", p.name)
        return int(m.group(1)) if m else p.name

    return sorted(imgs, key=key_fn)


def eval_one_query_one_image(
    preds: List[PredItem],
    gt_dir: Path,
    image_name: str,
    q: int,
    cls_thr: float,
    mask_thr: float,
    iou_thr: float,
    resize_to: int,
    pred_bin_thr: int,
) -> Tuple[int, Dict]:
    """Return value in {0,1,2} and debug dict."""

    candidates = [p for p in preds if p.q == q and (p.cls_conf > cls_thr) and (p.mask_conf > mask_thr) and p.label > 0]
    dbg = {
        "q": q,
        "n_total": sum(1 for p in preds if p.q == q),
        "n_candidates": len(candidates),
        "best_iou": None,
        "best_pred": None,
    }

    if not candidates:
        return 0, dbg

    best_iou = -1.0
    any_correct = False

    for cand in candidates:
        gt_path = gt_dir / image_name / f"{cand.label}.png"
        if not gt_path.exists():
            # cannot verify -> wrong for this candidate
            continue

        pred01 = read_mask_as_binary(cand.path, threshold=pred_bin_thr)
        gt01 = read_mask_as_binary(gt_path, threshold=0)
       
        pred01 = resize_bin(pred01, size=resize_to)
        gt01 = resize_bin(gt01, size=resize_to)
        # cv2.imwrite(f"/data/yjh_files/code/tools/pred_q{q}.png", pred01 * 255)
        # cv2.imwrite(f"/data/yjh_files/code/tools/gt_q{q}.png", gt01 *225)
        print(pred01.max(), gt01.max())
        val = iou(pred01, gt01)

        if val > best_iou:
            best_iou = val
            dbg["best_pred"] = cand.path.name
            dbg["best_iou"] = float(val)

        if val > iou_thr:
            any_correct = True
            # You only need one correct.
            break

    if any_correct:
        return 1, dbg
    # If we had candidates but none could be verified as correct (either low iou or missing GT)
    # treat as wrong.
    if best_iou >= 0:
        dbg["best_iou"] = float(best_iou)
    return 2, dbg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", type=str, default="/data/yjh_files/indice/baseline_2/")
    ap.add_argument("--gt_dir", type=str, default="/data/yjh_files/indice/gt")
    ap.add_argument("--seq_prefix", type=str, default="seq_2_")

    ap.add_argument("--cls_thr", type=float, default=0.85)
    ap.add_argument("--mask_thr", type=float, default=0.8)
    ap.add_argument("--iou_thr", type=float, default=0.75)
    ap.add_argument("--resize_to", type=int, default=128)
    ap.add_argument(
        "--pred_bin_thr",
        type=int,
        default=0,
        help="Prediction png binarization threshold: pixel>thr treated as foreground.",
    )

    ap.add_argument(
        "--out_npy",
        type=str,
        default="/data/yjh_files/indice/baseline_2/seq_2_query_matrix.npy",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default="/data/yjh_files/indice/baseline_2/seq_2_query_matrix.csv",
    )
    ap.add_argument(
        "--out_debug_json",
        type=str,
        default="/data/yjh_files/indice/baseline_2/seq_2_query_matrix_debug.json",
    )
    args = ap.parse_args()

    baseline_dir = Path(args.baseline_dir)
    gt_dir = Path(args.gt_dir)
    out_npy = Path(args.out_npy)
    out_csv = Path(args.out_csv)
    out_debug_json = Path(args.out_debug_json)
    out_npy.parent.mkdir(parents=True, exist_ok=True)

    img_dirs = list_seq2_images(baseline_dir, seq_prefix=args.seq_prefix)
    if not img_dirs:
        raise FileNotFoundError(f"No image dirs found under {baseline_dir} with prefix {args.seq_prefix}")

    N = len(img_dirs)
    mat = np.zeros((10, N), dtype=np.int64)
    debug: Dict[str, Dict] = {}

    for j, img_dir in enumerate(tqdm(img_dirs, desc="eval_images", unit="img")):
        image_name = img_dir.name
        debug[image_name] = {}

        # parse all predictions in this image dir
        preds: List[PredItem] = []
        for p in img_dir.iterdir():
            if p.suffix.lower() != ".png":
                continue
            item = parse_pred_filename(p)
            if item is None:
                continue
            # keep only q in 0..9
            if 0 <= item.q <= 9:
                preds.append(item)

        for q in range(10):
            val, dbg = eval_one_query_one_image(
                preds=preds,
                gt_dir=gt_dir,
                image_name=image_name,
                q=q,
                cls_thr=args.cls_thr,
                mask_thr=args.mask_thr,
                iou_thr=args.iou_thr,
                resize_to=args.resize_to,
                pred_bin_thr=args.pred_bin_thr,
            )
            mat[q, j] = val
            debug[image_name][str(q)] = dbg

    np.save(out_npy, mat)

    # Write CSV: rows are queries, columns are images
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        header = ["query"] + [d.name for d in img_dirs]
        w.writerow(header)
        for q in range(10):
            w.writerow([q] + mat[q, :].tolist())

    with out_debug_json.open("w") as f:
        json.dump(
            {
                "shape": list(mat.shape),
                "baseline_dir": str(baseline_dir),
                "gt_dir": str(gt_dir),
                "seq_prefix": args.seq_prefix,
                "thresholds": {
                    "cls_thr": args.cls_thr,
                    "mask_thr": args.mask_thr,
                    "iou_thr": args.iou_thr,
                    "resize_to": args.resize_to,
                    "pred_bin_thr": args.pred_bin_thr,
                },
                "debug": debug,
            },
            f,
            indent=2,
        )

    print(f"[DONE] matrix saved: {out_npy} shape={mat.shape}")
    print(f"[DONE] csv saved:    {out_csv}")
    print(f"[DONE] debug saved:  {out_debug_json}")


if __name__ == "__main__":
    main()
