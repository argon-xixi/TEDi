"""Overlay EndoVis2017 **predicted** PNG mask on GT h5 image.

你现在的预测 mask 已经导出为 PNG（像素值 0..7），例如：
  /data/yjh_files/code/Mask2Former-Simplify-master/endovis_2017_baseline_512_fold3_train/seq_1_frame200.png

该脚本会：
1) 遍历 pred_dir 下所有 PNG
2) 根据 PNG 文件名（stem）在 gt_h5_dir 中找到同名 h5（stem.h5）
3) 从 h5 读取 image_left（或 image），将 image resize 到 mask 尺寸
4) 把 **预测 mask** overlay 到 image 上并保存

Usage:
  python tools/visualize_endovis2017_h5_overlay_pred.py \
    --pred_dir /data/yjh_files/code/Mask2Former-Simplify-master/endovis_2017_baseline_512_fold3_train \
    --gt_h5_dir /data/yjh_files/data/Endovis2017/data_h5/train \
    --out_dir /data/yjh_files/code/Mask2Former-Simplify-master/endovis2017_overlay_pred \
    --alpha 0.6
"""

import argparse
import glob
import os
from typing import Iterable, Tuple

import cv2
import h5py
import numpy as np


def _read_png_mask(path: str) -> np.ndarray:
    """读预测 mask PNG 为 HxW int64（支持灰度/调色板/RGB）。"""
    from PIL import Image

    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.int64)


def read_h5_image(path: str, img_key_priority: Tuple[str, ...] = ("image_left", "image")) -> np.ndarray:
    """读取 h5 中的原图（BGR, uint8, HxWx3）。"""
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
    return img

def _apply_color_overlay(bgr_img: np.ndarray, label_mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay a label mask on BGR image.

    label_mask: HxW, values in {0..7}, where 0=background
    """
    # BGR colors (OpenCV)
    # 1..7 correspond to EndoVis2017 categories order in README.
    colors = {

        1: (226, 185, 5), 
        2: (39, 151, 187),      
        3: (69, 179,84 ),     
        4: (151, 181,50 ),    
        5: (72,203,137),    
        6: (191, 131,137 ),    
        7: (162, 109,199 ), 
    }

    # 注意：保持输出为 BGR，方便 cv2.imwrite
    out = bgr_img.copy()
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)  # 转到 RGB 方便计算，最后再转回 BGR 保存
    for cid, color in colors.items():
        m = (label_mask == cid)
        if not np.any(m):
            continue
        overlay = np.zeros_like(out, dtype=np.uint8)
        overlay[:, :] = color
        out[m] = (out[m] * (1 - alpha) + overlay[m] * alpha).astype(np.uint8)
    return out

def _find_pred_pngs(pred_dir: str, pred_glob: str = "*.png") -> Iterable[str]:
    return sorted(glob.glob(os.path.join(pred_dir, pred_glob)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pred_dir",
        type=str,
        default="/data/yjh_files/code/Mask2Former-Simplify-master/endovis_2017_baseline_512_fold2_val",
        help="预测 mask PNG 目录",
    )
    p.add_argument(
        "--gt_h5_dir",
        type=str,
        default="/data/yjh_files/data/Endovis2017/data_h5/train",
        help="GT h5 目录（用 PNG 文件名在此目录下找同名 h5）",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="/data/yjh_files/code/Mask2Former-Simplify-master/endovis2017_overlay_pred",
        help="输出目录",
    )
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--limit", type=int, default=-1, help="Process only first N files (debug)")
    p.add_argument("--pred_glob", type=str, default="*.png", help="预测 PNG 的 glob（默认只扫一层）")
    p.add_argument("--img_keys", type=str, default="image_left,image", help="h5 图像 key 优先级，用逗号分隔")
    p.add_argument("--ext", type=str, default=".png", choices=[".png", ".jpg", ".jpeg"])  # output format
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "overlay"), exist_ok=True)

    pred_pngs = list(_find_pred_pngs(args.pred_dir, pred_glob=args.pred_glob))
    if args.limit and args.limit > 0:
        pred_pngs = pred_pngs[: args.limit]
    if not pred_pngs:
        raise FileNotFoundError(f"No prediction png under {args.pred_dir} (glob={args.pred_glob})")

    img_key_priority = tuple([s.strip() for s in args.img_keys.split(",") if s.strip()])

    missing = 0
    for i, pred_path in enumerate(pred_pngs):
        stem = os.path.splitext(os.path.basename(pred_path))[0]
        gt_h5_path = os.path.join(args.gt_h5_dir, f"{stem}.h5")
        if not os.path.exists(gt_h5_path):
            missing += 1
            continue
        # if "seq_8_frame039" not in pred_path:
        #     continue

        pred_mask = _read_png_mask(pred_path)  # HxW
        img_bgr = read_h5_image(gt_h5_path, img_key_priority=img_key_priority)
        # print(np.unique(pred_mask), pred_mask.shape, img_bgr.shape)
        # # # resize image 到 mask 尺寸
        # pred_mask[pred_mask == 5 ]= 0
        h, w = int(pred_mask.shape[0]), int(pred_mask.shape[1])
        if img_bgr.shape[0] != h or img_bgr.shape[1] != w:
            img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)

        over = _apply_color_overlay(img_bgr, pred_mask, alpha=float(args.alpha))
        # over = cv2.cvtColor(over, cv2.COLOR_BGR2RGB)  # 转回 RGB 保存
        out_overlay = os.path.join(args.out_dir, "overlay", f"{stem}{args.ext}")
        cv2.imwrite(out_overlay, over)

        if (i + 1) % 200 == 0:
            print(f"[{i+1}/{len(pred_pngs)}] saved: {out_overlay}")

    print(f"Done. Saved overlays to {os.path.join(args.out_dir, 'overlay')}. Missing h5: {missing}")


if __name__ == "__main__":
    main()
