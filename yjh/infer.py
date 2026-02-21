#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
import glob
import argparse
import random

import numpy as np
import torch

# 按照 test copy.py 的写法，保留对原项目的路径设置（运行环境中如有需要可自行调整）
sys.path.append('/home/yjh/code_yjh_bishe/Mask2Former-Simplify-master')

from fvcore.common.config import CfgNode
from configs.config import Config
from tensorboardX import SummaryWriter

from Data import dataloaders
from maskformer_train_infer import MaskFormer_baseline_infer


# -----------------------------------------------------------------------------
# 与 test copy.py 保持一致的数据过滤列表（EndoVis2018 用）
# -----------------------------------------------------------------------------

remove_list_train1 = []
for index in range(119, 149):
    remove_list_train1.append('seq_3_frame' + str(index))

remove_list_val1 = []
path_2017_test_label = '/data1/yuanjiahong_files/bishe/EndoVis2017/data_raw/test_label/'
if os.path.exists(path_2017_test_label):
    for k in os.listdir(path_2017_test_label):
        if 'seq_2_' in k:
            remove_list_val1.append(k.split('.')[0])

remove_list = {
    'seq_2_frame004', 'seq_4_frame137', 'seq_6_frame104', 'seq_6_frame138',
    'seq_10_frame072', 'seq_10_frame105', 'seq_15_frame139',
    'seq_16_frame030', 'seq_16_frame068'
}

remove_list_val = {
    'seq_2_frame107', 'seq_2_frame108', 'seq_2_frame109', 'seq_2_frame110',
    'seq_2_frame111', 'seq_2_frame112', 'seq_2_frame113', 'seq_2_frame114',
    'seq_2_frame115', 'seq_2_frame116', 'seq_2_frame117', 'seq_2_frame118',
    'seq_2_frame119', 'seq_9_frame50', 'seq_9_frame57', 'seq_9_frame70',
    'seq_9_frame71', 'seq_9_frame72', 'seq_9_frame73', 'seq_9_frame74',
    'seq_9_frame75', 'seq_9_frame76', 'seq_9_frame140'
}



class MaskFormer_infer:
    """专门用于推理与评估的封装类。

    只依赖：
    - test.py 中已有的 `MaskFormer_baseline`（内部含 evaluate 等所有指标计算逻辑）
    - Data.dataloaders.get_dataloaders（与训练时保持一致的数据加载方式）
    """

    def __init__(self, cfg):
        # 强制设置为推理模式，便于在 MaskFormer_baseline.__init__ 中走 INFER_PRETRAINED_WEIGHTS 分支
        cfg.inferonly = True
        self.cfg = cfg
        self.baseline = MaskFormer_baseline_infer(cfg)

    def _build_eval_loader(self, train_paths_temp, test_paths_temp):
        """根据 cfg.dataset 构造 evaluation dataloader。

        与 MaskFormer_baseline.train 中的逻辑保持一致，只是这里只需要 val_dataloader/eval_loader。
        """

        cfg = self.cfg

        if cfg.dataset == 'EndoVis2017':
            # 与 MaskFormer_baseline.train 中的划分一致：dataset_6 / dataset_7 作为验证
            TrainPath = []
            ValPath = []
            for p in train_paths_temp:
                if 'dataset_6' in p or 'dataset_7' in p:
                    ValPath.append(p)
                else:
                    TrainPath.append(p)

            train_loader, eval_loader = dataloaders.get_dataloaders(
                TrainPath,
                ValPath,
                cfg,
                batch_size=cfg.batch_size,
                num_workers=cfg.workers,
                version=0,
                bina=cfg.bina,
            )

        elif cfg.dataset == 'EndoVis2018':
            # 直接使用 train_paths_temp / test_paths_temp，与 test copy.py 中保持一致
            train_loader, eval_loader = dataloaders.get_dataloaders(
                train_paths_temp,
                test_paths_temp,
                cfg,
                batch_size=cfg.batch_size,
                num_workers=cfg.workers,
                version=0,
                bina=cfg.bina,
            )
        else:
            raise ValueError(f"不支持的数据集类型: {cfg.dataset}")

        return eval_loader

    def infer(self, train_paths_temp, test_paths_temp):
        """执行一次完整的推理与评估流程。

        * 构造 eval_loader；
        * 创建与训练阶段相同层级的日志目录和 SummaryWriter；
        * 调用 `MaskFormer_baseline.evaluate`，在其中完成：
            - 前向推理
            - IoU / Dice / mIoU 等所有指标计算
            - TensorBoard 标量记录

        返回：
            一个简单的 dict，其中包含 evaluate 返回的主评分（score）。
        """

        cfg = self.cfg
        eval_loader = self._build_eval_loader(train_paths_temp, test_paths_temp)

        # 日志目录：延续 test copy.py 的写法，但在 task 名后加一个 "_infer" 后缀以示区分
        if cfg.dataset == 'EndoVis2017':
            base_dir = '/data1/yuanjiahong_files/bishe/EndoVis2017/result_new/'
        elif cfg.dataset == 'EndoVis2018':
            base_dir = '/data1/yuanjiahong_files/bishe/EndoVis2018/result_new/'
        else:
            base_dir = './results/'

        writer_dir = os.path.join(base_dir, cfg.task + '_infer')
        os.makedirs(writer_dir, exist_ok=True)
        writer = SummaryWriter(writer_dir, flush_secs=15)

        # 这里 epoch 固定为 0，仅用于日志 x 轴标记
        score = self.baseline.evaluate(eval_loader, epoch=0, writer=writer)

        writer.close()

        return {"main_score": score}




def get_args():
    parser = argparse.ArgumentParser(description="Infer/Validate Mask2Former on specified dataset")

    # 与 test copy.py 保持同名参数，只是将 inferonly 默认设为 True，更符合本文件用途
    parser.add_argument('--inferonly', default=True, type=bool, help='是否只执行推理')
    parser.add_argument("--sam", default=False, type=bool, help='是否运行 SAM 模块')
    parser.add_argument("--bina", default=False, type=bool, help='是否运行 bina 模块')
    parser.add_argument("--bbox", default=False, type=bool, help='是否运行 bbox 模块')
    parser.add_argument("--sync_bn", default=False, type=bool, help='是否使用 SyncBN')
    parser.add_argument(
        '--config',
        type=str,
        default='/home/yjh/code_yjh_bishe/Mask2Former-Simplify-master/configs/maskformer_yjh.yaml',
        help='配置文件路径 (yaml)'
    )
    parser.add_argument('--tracker', default=False, type=bool, help='是否使用 track 模块')
    parser.add_argument('--output_feature', default=False, type=bool, help='是否输出特征')
    parser.add_argument('--refiner', default=False, type=bool, help='是否使用 refine 模块')

    # DDP / 多卡参数，保留接口但本脚本默认不强依赖
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed inference")
    parser.add_argument("--DDP", type=str, default=False, help="Whether to use DistributedDataParallel")
    parser.add_argument("--ngpus", default=8, type=int, help="最大 GPU 数（实际以 torch.cuda.device_count 为准）")

    parser.add_argument("--project_name", default='NuImages_swin_base_Seg', type=str)
    parser.add_argument("--resume", default=False, type=bool)
    parser.add_argument("--dataset", type=str, default="EndoVis2018", choices=['EndoVis2017', 'EndoVis2018'])
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--task", type=str, default="endovis_2018_baseline_swins_10_0")
    parser.add_argument(
        "--root",
        type=str,
        default="/data1/yuanjiahong_files/bishe/EndoVis2018/data_h5_baseline/",
        help="训练/测试数据集根目录（EndoVis2018 使用）",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=4, dest="batch_size")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=6e-5, dest="lr")

    parser.add_argument("--learning-rate-scheduler", type=str, default="true", dest="lrs")
    parser.add_argument("--learning-rate-scheduler-minimum", type=float, default=1e-3, dest="lrs_min")
    parser.add_argument("--multi-gpu", type=str, default="true", dest="mgpu", choices=["true", "false"])

    args = parser.parse_args()

    # local_rank 可从环境变量中覆盖
    args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))

    # 载入 yaml 配置，并与命令行参数合并（与 test copy.py 一致）
    cfg_base = CfgNode.load_yaml_with_base(args.config, allow_unsafe=True)
    cfg = cfg_base
    for k, v in args.__dict__.items():
        cfg[k] = v
    cfg = Config(cfg)

    # 实际可用 GPU 数
    cfg.ngpus = torch.cuda.device_count()

    return cfg



def build(cfg):
    """按照 test copy.py 的方式构建 train/test 路径列表，然后调用 MaskFormer_infer 完成评估。"""

    train_paths_temp = []
    test_paths_temp = []

    if cfg.dataset == 'EndoVis2017':
        # 与 test copy.py 保持一致
        train_path = '/data1/yuanjiahong_files/bishe/EndoVis2017/data_raw/cropped_train_left'
        train_h5_ori = "/data1/yuanjiahong_files/bishe/EndoVis2017/data_h5_bbox/train/"

        if os.path.exists(train_path):
            for seq in os.listdir(train_path):
                seq_dir = os.path.join(train_path, seq)
                if os.path.isdir(seq_dir):
                    img_dir = os.path.join(seq_dir, 'images')
                    if not os.path.exists(img_dir):
                        continue
                    for img_name in os.listdir(img_dir):
                        if not img_name.lower().endswith('.jpg'):
                            continue
                        name = seq + "_" + img_name.replace('.jpg', '')
                        train_paths_temp.append(os.path.join(train_h5_ori, name))

        test_paths = sorted(glob.glob(os.path.join(
            "/data1/yuanjiahong_files/bishe/EndoVis2017/data_h5_bina/", 'test', '*')
        ))
        for p in test_paths:
            test_paths_temp.append(p.split('.')[0])

    elif cfg.dataset == 'EndoVis2018':
        train_path = '/data1/yuanjiahong_files/bishe/EndoVis2018/data_raw/train/images/'
        train_h5_ori = "/data1/yuanjiahong_files/bishe/EndoVis2018/data_h5_baseline/train/"

        if os.path.exists(train_path):
            for img_name in os.listdir(train_path):
                stem = img_name.replace('.png', '')
                if stem in remove_list or stem in remove_list_train1:
                    continue
                name = stem
                train_paths_temp.append(os.path.join(train_h5_ori, name))

        test_paths = sorted(glob.glob(os.path.join(cfg.root, 'test', '*')))
        for p in test_paths:
            stem = os.path.basename(p).replace('.h5', '')
            if stem in remove_list or stem in remove_list_val:
                continue
            test_paths_temp.append(p.split('.')[0])

    else:
        raise ValueError(f"不支持的数据集类型: {cfg.dataset}")

    infer_engine = MaskFormer_infer(cfg)
    metrics = infer_engine.infer(train_paths_temp, test_paths_temp)
    return metrics


def main():
    cfg = get_args()

    # 固定随机种子，保证结果可复现（与 test copy.py 一致）
    seed = int(cfg.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    metrics = build(cfg)
    print("Inference finished. Main score:", metrics.get('main_score'))


if __name__ == "__main__":
    main()

