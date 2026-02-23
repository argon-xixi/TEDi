#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""离线评估脚本：pred PNG vs gt h5("mask").

用法示例：

python Mask2Former-Simplify-master/tools/offline_eval_png_vs_h5.py \
  --pred_root /path/to/pred_png_dir \
  --gt_root /path/to/gt_h5_dir \
  --num_classes 7 \
  --iou_thresh 0.8 \
  --save_dir /path/to/save_metrics

说明：
- 会递归扫描 pred_root 下所有 png（可用 --pred_glob 自定义）
- GT 在 gt_root 下与 pred 保持相同相对路径、同名（扩展名为 .h5/.hdf5）
"""

import argparse


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_root", type=str, required=True, help="预测 PNG 掩膜目录（根目录）")
    p.add_argument("--gt_root", type=str, required=True, help="标签 h5 目录（根目录）")
    p.add_argument("--num_classes", type=int, default=7)
    p.add_argument("--gt_key", type=str, default="mask")
    p.add_argument("--iou_thresh", type=float, default=0.8)
    p.add_argument("--pred_glob", type=str, default="**/*.png")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--save_dir", type=str, default=None, help="保存混淆矩阵/AUC图的目录；不填则不保存")
    p.add_argument("--epoch", type=int, default=0)
    p.add_argument("--strict_match", action="store_true", help="严格匹配：找不到 GT 直接报错")
    p.add_argument("--no_save_fig", action="store_true", help="不保存图（即使提供了 save_dir）")
    p.add_argument("--quiet", action="store_true")
    return p


def main():
    args = build_parser().parse_args()

    # 保证能 import 到工程
    # 该脚本位于 Mask2Former-Simplify-master/tools 下，因此把上一级加入 sys.path
    import os
    import sys

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from yjh.utils_yjh import offline_eval_pred_png_vs_gt_h5

    metrics = offline_eval_pred_png_vs_gt_h5(
        pred_root=args.pred_root,
        gt_root=args.gt_root,
        num_classes=args.num_classes,
        gt_key=args.gt_key,
        iou_thresh=args.iou_thresh,
        pred_glob=args.pred_glob,
        strict_match=args.strict_match,
        device=args.device,
        save_dir=args.save_dir,
        epoch=args.epoch,
        save_fig=(not args.no_save_fig),
        verbose=(not args.quiet),
    )

    if args.quiet:
        # quiet 模式仍输出一个最关键的 summary，方便脚本被 bash 调用
        print(
            f"N={metrics['num_samples']} bin_iou={metrics['bin_iou_mean']:.4f} bin_dice={metrics['bin_dice_mean']:.4f} "
            f"binary_dice={metrics['binary_dice']:.4f} auc_macro={metrics['auc_macro']:.4f} auc_micro={metrics['auc_micro']:.4f}"
        )


if __name__ == "__main__":
    main()
