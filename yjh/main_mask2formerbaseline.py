import sys
sys.path.append('/home/yjh/code_yjh_bishe/Mask2Former-Simplify-master')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7,0"

import torch

print("haha")
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
# # 设置主进程的 IP 地址和端口号
# rank, local_rank, world_size = init_distributed_mode()
# cfg.LOCAL_RANK = local_rank 
# os.environ['MASTER_ADDR'] = '166.111.72.108'
# os.environ['MASTER_PORT'] = '29501'
# os.environ['RANK'] = '0's
# os.environ['WORLD_SIZE'] = '1'
# RANK = int(os.environ['SLURM_PROCID'])  # 进程序号，用于进程间通信
# LOCAL_RANK = int(os.environ['SLURM_LOCALID']) # 本地设备序号，用于设备分配.
# GPU_NUM = int(os.environ['SLURM_NTASKS'])     # 使用的 GPU 总数.
# RANK = int(os.environ.get("SLURM_PROCID", "0"))
# LOCAL_RANK = int(os.environ.get("SLURM_LOCALID", "0"))
# GPU_NUM = int(os.environ.get("SLURM_NTASKS", "1"))

# IP = os.environ['SLURM_STEP_NODELIST'] #进程节点 IP 信息.
# BATCH_SIZE = 16  # 单张 GPU 的大小.

# kill僵尸进程
# import os
# result = os.popen("fuser -v /dev/nvidia*").read()
# results = result.split()
# for pid in results:
#     os.system(f"kill -9 {int(pid)}")
    
# import torch
# import torchvision
# print(torch.__version__, torchvision.__version__) 
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())

from pathlib import Path
import argparse
import time
import numpy as np
import glob
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import random
import torch, gc

# gc.collect()

# torch.cuda.empty_cache()

import torch.nn as nn
from fvcore.common.config import CfgNode
from configs.config import Config
import utils
from Data import dataloaders

from Metrics import performance_metrics
from Metrics import losses
from utils import *
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from Metrics.mIOU_new import eval_endovis
from Metrics.dice_onehot import dice_coeff
from maskformer_train_sam_DP import MaskFormer
# from maskformer_train_track import MaskFormer_track
from maskformer_train_track import MaskFormer_track
from maskformer_train_bbox import MaskFormer_bbox
from maskformer_train_baseline import MaskFormer_baseline
from maskformer_train_bbox_DP import MaskFormer_bbox
from maskformer_train_feature_output import MaskFormer_feature_output
from maskformer_train_refine import MaskFormer_refine
from infer import MaskFormer_infer
from detectron2.config import CfgNode as CN
from maskformer_train_sam_new import MaskFormer_sam_new
remove_list_train1=[]
for index in range(119,149):
    remove_list_train1.append('seq_3_frame'+str(index))
remove_list_val1=[]
path='/data1/yuanjiahong_files/bishe/EndoVis2017/data_raw/test_label/'
for k in os.listdir(path):
    if 'seq_2_'in k:
        remove_list_val1.append(k.split('.')[0])
remove_list={'seq_2_frame004','seq_4_frame137','seq_6_frame104','seq_6_frame138','seq_10_frame072','seq_10_frame105','seq_15_frame139',
             'seq_16_frame030','seq_16_frame068'}
remove_list_val={'seq_2_frame107', 'seq_2_frame108', 'seq_2_frame109', 'seq_2_frame110', 'seq_2_frame111', 'seq_2_frame112',
 'seq_2_frame113', 'seq_2_frame114', 'seq_2_frame115', 'seq_2_frame116', 'seq_2_frame117', 'seq_2_frame118', 'seq_2_frame119','seq_9_frame50','seq_9_frame57','seq_9_frame70','seq_9_frame71','seq_9_frame72','seq_9_frame73','seq_9_frame74','seq_9_frame75','seq_9_frame76','seq_9_frame140'}
# remove_list_train={'seq_11_frame012', 'seq_3_frame019', 'seq_11_frame077', 'seq_11_frame013', 'seq_7_frame109', 'seq_11_frame066',
#              'seq_11_frame067', 'seq_11_frame071', 'seq_11_frame076', 'seq_11_frame014', 'seq_3_frame020', 'seq_11_frame075'}
# remove_list_val={'seq_2_frame114', 'seq_2_frame108', 'seq_9_frame071', 'seq_2_frame107', 'seq_9_frame050', 'seq_9_frame140', 'seq_2_frame116',
#  'seq_9_frame057', 'seq_9_frame073', 'seq_2_frame117', 'seq_9_frame075', 'seq_2_frame111', 'seq_9_frame072', 'seq_2_frame118',
#  'seq_2_frame112', 'seq_9_frame070', 'seq_2_frame113', 'seq_9_frame074', 'seq_9_frame076', 'seq_2_frame110', 'seq_2_frame115', 'seq_2_frame109'}

def dist_init(host_addr, rank, local_rank, world_size, port=23456):
    host_addr_full = 'tcp://' + host_addr + ':' + str(port)
    torch.distributed.init_process_group("nccl", init_method=host_addr_full,
                                         rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    assert torch.distributed.is_initialized()
    
def build(cfg):

    train_paths_temp=[]
    test_paths_temp=[]
    
    if cfg.dataset == 'EndoVis2017':
        writer_dir = os.path.join('/data1/yuanjiahong_files/bishe/EndoVis2017/result_new/', cfg.task)
        if not os.path.exists(writer_dir):
            os.makedirs(writer_dir)
        # os.mkdir(writer_dir)只能创建单级目录
        writer = SummaryWriter(writer_dir, flush_secs=15)
        train_path='/data1/yuanjiahong_files/bishe/EndoVis2017/data_raw/cropped_train_left'
        train_h5_ori="/data1/yuanjiahong_files/bishe/EndoVis2017/data_h5_bbox/train/"
    
        for i in os.listdir(train_path):
            if os.path.isdir(os.path.join(train_path,i)):
                path1=os.path.join(train_path,i,'images')
                for k in os.listdir(path1):
                    name=i+"_"+k.replace('.jpg','')
                    train_paths_temp.append(os.path.join(train_h5_ori,name))
        test_paths=sorted(glob.glob(os.path.join("/data1/yuanjiahong_files/bishe/EndoVis2017/data_h5_bina/",'test','*')))
        for i in test_paths:
            test_paths_temp.append(i.split('.')[0])
            
    if cfg.dataset == 'EndoVis2018':
        writer_dir = os.path.join('/data1/yuanjiahong_files/bishe/EndoVis2018/result_new/', cfg.task)
        if not os.path.exists(writer_dir):
            os.makedirs(writer_dir)
            # os.mkdir(writer_dir)只能创建单级目录
        writer = SummaryWriter(writer_dir, flush_secs=15)
        train_path='/data1/yuanjiahong_files/bishe/EndoVis2018/data_raw/train/images/'
        train_h5_ori="/data1/yuanjiahong_files/bishe/EndoVis2018/data_h5_baseline/train/"
        for i in os.listdir(train_path):
            if i.replace('.png','') in remove_list or i.replace('.png','') in remove_list_train1:
                continue
            name=i.replace('.png','')
            train_paths_temp.append(os.path.join(train_h5_ori,name))
        test_paths=sorted(glob.glob(os.path.join(cfg.root,'test','*')))
        for i in test_paths:
            if i.split('/')[-1].replace('.h5','') in remove_list  or i.split('/')[-1].replace('.h5','') in remove_list_val :
            # if i.split('/')[-1].replace('.h5','') in remove_list  :
                continue
            # if 'seq_2_' not in i :
            #         continue
            test_paths_temp.append(i.split('.')[0])
    # if cfg.inferonly:
    #     seg_model = MaskFormer_infer(cfg)
    #     seg_model.infer(train_paths_temp, test_paths_temp)
    # print(train_paths_temp)
    # if cfg.output_feature:
    #     seg_model = MaskFormer_feature_output(cfg)
    #     cfg.epochs=2
    #     seg_model.train(train_paths_temp, test_paths_temp, cfg.epochs,writer)
    #     exit()
    if cfg.refiner:
        cfg.SOLVER
        cfg.SOLVER.MAX_ITER= 20000
        cfg.MODEL.TRACKER = CN()
        cfg.MODEL.REFINER = CN()
        seg_model = MaskFormer_refine(cfg)
        seg_model.train(train_paths_temp, test_paths_temp, cfg.epochs,writer)
    if cfg.tracker:
        cfg.SOLVER
        cfg.SOLVER.MAX_ITER= 20000
        cfg.MODEL.TRACKER = CN()
        cfg.MODEL.REFINER = CN()
        cfg.MODEL.REFINER.DECODER_LAYERS = 6
        seg_model = MaskFormer_track(cfg)
        seg_model.train(train_paths_temp, test_paths_temp, cfg.epochs,writer)
    
    if not cfg.sam :
        seg_model = MaskFormer_baseline(cfg)
        seg_model.train(train_paths_temp, test_paths_temp, cfg.epochs,writer)
    if cfg.bbox :
        seg_model = MaskFormer_bbox(cfg)
        seg_model.train(train_paths_temp, test_paths_temp, cfg.epochs,writer)
    else:
        # seg_model = MaskFormer(cfg)
        # seg_model.train(train_paths_temp, test_paths_temp, cfg.epochs,writer)
        seg_model = MaskFormer_sam_new(cfg)
        seg_model.train(train_paths_temp, test_paths_temp, cfg.epochs,writer)
def get_args():
    parser = argparse.ArgumentParser(description="Train FCBFormer on specified dataset")
    parser.add_argument('--inferonly', default=False,type=bool )  #是否推理
    parser.add_argument("--sam", default=False, type=bool) #是否运行sam模块
    parser.add_argument("--bina", default=False, type=bool) #是否运行bina模块
    parser.add_argument("--bbox", default=False, type=bool) #是否运行bbox模块 
    parser.add_argument("--sync_bn", default=False, type=bool) #是否使用sync_bn
    parser.add_argument('--config', type=str, default='/home/yjh/code_yjh_bishe/Mask2Former-Simplify-master/configs/maskformer_yjh.yaml')
    parser.add_argument('--tracker', default=False, type=bool) #是否使用track模块
    parser.add_argument('--output_feature', default=False, type=bool)
    parser.add_argument('--refiner', default=False, type=bool) #是否使用refine模块
    # 声明 local_rank 参数（DDP自动注入）
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    parser.add_argument("--DDP", type=str, default=False, help="World size for distributed training")
    parser.add_argument("--ngpus", default=8, type=int)
    parser.add_argument("--project_name", default='NuImages_swin_base_Seg', type=str)
    parser.add_argument("--resume", default=False, type=bool)
    parser.add_argument("--dataset", type=str, default="EndoVis2018", choices=['EndoVis2017','EndoVis2018']) #使用数据集
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--task", type=str, default="endovis_2018_baseline_swins_13_0") #任务名（保存文件夹名）
    parser.add_argument("--root", type=str, default="/data1/yuanjiahong_files/bishe/EndoVis2018/data_h5_baseline/") #训练数据集根目录
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16) #24
    parser.add_argument("--workers", type=int, default=16)  #16
    parser.add_argument("--learning-rate", type=float, default=6e-5, dest="lr")
    
    parser.add_argument(
        "--learning-rate-scheduler", type=str, default="true", dest="lrs"
    )
    parser.add_argument(
        "--learning-rate-scheduler-minimum", type=float, default=1e-3, dest="lrs_min"
    )
    parser.add_argument(
        "--multi-gpu", type=str, default="true", dest="mgpu", choices=["true", "false"]
    )
    args = parser.parse_args()
    args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    # cfg_yjh = Config.fromfile(args.config) #载入配置文件yaml
    cfg_base = CfgNode.load_yaml_with_base(args.config, allow_unsafe=True) #载入配置文件yaml
    # cfg_base.update(cfg_yjh.__dict__.items())
    cfg = cfg_base
    for k, v in args.__dict__.items():
        cfg[k] = v
    cfg = Config(cfg)
    cfg.ngpus = torch.cuda.device_count()
    # local_rank=os.getenv('LOCAL_RANK', -1)
    # cfg.LOCAL_RANK=local_rank
    # if torch.cuda.device_count() > 1:
    #     # cfg.LOCAL_RANK = torch.distributed.get_rank()
    #     cfg.LOCAL_RANK=os.getenv('LOCAL_RANK', -1)
    #     print("Using {} GPUs"   . format(torch.cuda.device_count()))
    #     print("LOCAL_RANK: {}".format(cfg.LOCAL_RANK))
    #     torch.cuda.set_device(cfg.LOCAL_RANK)
    # return cfg
    if cfg.DDP==True:
        if torch.cuda.device_count() > 1:
            dist.init_process_group(backend='nccl')
        cfg.ngpus = torch.cuda.device_count()
        if torch.cuda.device_count() > 1:
            cfg.local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(cfg.local_rank)
    # GPU_NUM = int(os.environ['SLURM_NTASKS'])     # 使用的 GPU 总数.
    # RANK = int(os.environ.get("SLURM_PROCID", "0"))
    # cfg.GPU_NUM = GPU_NUM
    # cfg.RANK = RANK
    return cfg
def main():
    config = get_args()

    seed = int(config.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    build(config)
if __name__ == "__main__":
    main()
# /data1/yuanjiahong_files/miniconda3/envs/yjh/bin/python /home/yjh/code_yjh_bishe/Mask2Former-Simplify-master/yjh/main_mask2formerbaseline.py
# /home/yjh/anaconda3/envs/yjh/bin/python -m torch.distributed.launch --nproc_per_node=4 /home/gjy/code_yjh/Mask2Former-Simplify-master/yjh/main_mask2former.py
# torchrun --nproc_per_node=4 --master_port=29500 /home/gjy/code_yjh/Mask2Former-Simplify-master/yjh/main_mask2former.py0
# 在经过数据清洗和数据增强后存入
# rsync -avP /data/guojiayi_files/yjh_data/ /data1/yuanjiahong_files/bishe/bishe/

