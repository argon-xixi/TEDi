#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   maskformer3D.py
@Time    :   2022/09/30 20:50:53
@Author  :   BQH 
@Version :   1.0
@Contact :   raogx.vip@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   DeformTransAtten分割网络训练代码
'''

# here put the import lib
import copy
import random
from statistics import mean
import torch
import numpy as np
import os
import time
import datetime
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch import distributed as dist
from torch.utils.data import DataLoader, SubsetRandomSampler
import sys
import math
import itertools
from PIL import Image
# import wandb
from Data import dataloaders

from modeling.MaskFormerModel_baseline import MaskFormerModel_baseline
from modeling.MaskFormerModel_bina import MaskFormerModel_bina
from utils.criterion import SetCriterion, Criterion
from utils.matcher import HungarianMatcher
from utils.summary import create_summary
from utils.solver import maybe_add_gradient_clipping
from utils.misc import load_parallal_model
# from dataset.NuImages import NuImages
from Segmentation import Segmentation
from yjh.mIOU_new import eval_endovis
import cv2
import pickle
from collections import OrderedDict
from yjh.utils_yjh import instance_inference_pure,batch_topk_instances_to_semantic_overwrite,visanddraw,semantic_inference_with_bg,compute_iou_and_dice,plot_query_semseg_distribution,overlay
# from modeling.memory.memory_encorder_classwise import ClasswiseQueryMemoryModule
import einops
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
import matplotlib
matplotlib.use('Agg')  # 服务器环境无显示器时避免报错
import matplotlib.pyplot as plt
#保存indice
def indice_dict_to_serializable(indice_dict):
    out = {}
    for k, v in indice_dict.items():
        out[k] = [
            (a.detach().cpu().tolist(), b.detach().cpu().tolist())
            for (a, b) in v
        ]
    return out
def append_epoch_jsonl(save_dir, epoch, step, name_list, indice_dict, extra=None):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"epoch_{epoch:04d}.jsonl")

    record = {
        "epoch": int(epoch),
        "step": int(step),
        "names": list(name_list),  # 你的 name list
        "indice_dict": indice_dict_to_serializable(indice_dict),
    }
    if extra is not None:
        record["extra"] = extra  # 你想加别的信息：loss, lr, etc.

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        
#保存cutmix反解码
def denorm_bchw(x, mean, std):
    """
    x: Tensor [B,C,H,W]
    mean/std: list/tuple 长度 C
    """
    mean = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    std  = torch.tensor(std,  device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return x * std + mean

def swap_first_last(x, k):
    x = x.clone()  # 不修改原 tensor
    idx_first = [slice(None)] * x.dim()
    idx_last = [slice(None)] * x.dim()
    idx_first[k] = 0
    idx_last[k] = -1
    # 交换
    tmp = x[tuple(idx_first)].clone()
    x[tuple(idx_first)] = x[tuple(idx_last)]
    x[tuple(idx_last)] = tmp
    return x
def reduce_value(value, average=True):
    world_size = dist.get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)   # 对不同设备之间的value求和
        if average:  # 如果需要求平均，获得多块GPU计算loss的均值
            value /= world_size

    return value
def get_segimg(semseg): #加上背景后argmax
    # 输入为pred_mask (bn,7,h,w)
    background_prob = 1 - semseg.sum(dim=1, keepdim=True)
    semseg_with_bg = torch.cat([background_prob, semseg], dim=1) 
    pred_mask=semseg_with_bg
    # pred_mask = semseg_with_bg.argmax(dim=1)
    return pred_mask #输出为(bn,h,w)
def dice_coeff(pred, target, num_cls=7, epsilon=1e-6):
    """
    pred:   (B, H, W)  预测标签，取值 0..7（含背景0）
    target: (B, H, W)  真实标签，取值 0..7（含背景0）
    num_cls: 非背景类别数（这里为7，因此总类= num_cls+1 = 8）
    """
    # 安全校验（GPU 友好）
    assert pred.shape == target.shape, "pred/target 形状不一致"
    assert target.max().item() <= num_cls and target.min().item() >= 0, "Target 越界"
    assert pred.max().item()   <= num_cls and pred.min().item()   >= 0, "Pred 越界"

    C = num_cls + 1  # 含背景的总类数

    # one-hot (B,C,H,W) 
    pred_one_hot   = F.one_hot(pred.long(),   num_classes=C).permute(0,3,1,2).float()
    target_one_hot = F.one_hot(target.long(), num_classes=C).permute(0,3,1,2).float()

    # 按类聚合（跨 batch、H、W）
    dims = (0, 2, 3)
    intersection = torch.sum(pred_one_hot * target_one_hot, dim=dims)          # (C,)
    union        = torch.sum(pred_one_hot, dim=dims) + torch.sum(target_one_hot, dim=dims)
    dice_scores  = (2.0 * intersection + epsilon) / (union + epsilon)          # (C,)

    # 仅统计前景类；并只对“目标中出现过的类”取平均，避免缺失类被当成1
    foreground = slice(1, None)  # 1..num_cls
    present = (torch.sum(target_one_hot, dim=dims) > 0)                         # (C,)
    present_fg = present[foreground]
    dice_fg = dice_scores[foreground]

    if present_fg.any():
        mean_dice = (dice_fg * present_fg.float()).sum() / (present_fg.float().sum())
    else:
        mean_dice = torch.tensor(0.0, device=dice_fg.device)

    return mean_dice
# def dice_coeff(pred, target, num_cls=7, epsilon=1e-6): #hard dice 更贴近实际，在计算loss时一般采用soft dice
 
#     """
#     pred: (batch_size, num_cls, H, W)  # 模型输出,不含背景类(类别1~7)
#     target: (batch_size, H, W)         # 真实标签,包含背景0类(0~7)
#     num_cls: 非背景类别数(此处为7)
#     """
#     # 确保 target 的类别范围合法
#     assert target.max() <= num_cls, "Target 包含超出模型预测能力的类别"
#     # get_label(pred,target)
   
#     # pred_class = torch.argmax(pred, dim=1)          # (batch, H, W) 值为 0~6
#                                    # 对齐到 target 的类别（1~7）
#     pred_class = pred.long()
#     pred_one_hot = F.one_hot(pred_class, num_classes=num_cls+1)  # (batch, H, W, num_cls+1)
#     pred_one_hot = pred_one_hot.permute(0, 3, 1, 2).float()[:,:,:,:]      # (batch, num_cls+1, H, W)
#     # (batch, num_cls+1, H, W)
#     # pred_one_hot = pred
#     # 将 target 转为 one-hot（含背景类0）
#     target_one_hot = F.one_hot(target.long(), num_classes=num_cls+1)  # (batch, H, W, num_cls+1)
#     target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()[:,:,:,:]      # (batch, num_cls+1, H, W)
#     # 计算各维度求和
#     intersection = torch.sum(pred_one_hot * target_one_hot, dim=(0, 2, 3))  # shape (num_cls+1,)
#     union = torch.sum(pred_one_hot, dim=(0, 2, 3)) + torch.sum(target_one_hot, dim=(0, 2, 3))

#     # 计算 Dice（排除背景0类）
#     dice_scores = (2.0 * intersection + epsilon) / (union + epsilon)
#     return dice_scores[1:].mean()  # 只取类别1~7的平均值

class MaskFormer_baseline():
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_queries = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        self.size_divisibility = cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY
        self.num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        # self.device = torch.device("cuda", cfg.LOCAL_RANK)
        # self.is_training = cfg.MODEL.IS_TRAINING
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.last_lr = cfg.SOLVER.LR
        self.start_epoch = 0
 
        # 记录当前使用的 query 策略：original / topk / linear
        # 对应 modeling/MaskFormerModel_baseline.py 中的 cfg.MODEL.MASK_FORMER.QUERY_VARIANT
        # 如果配置中没有该字段，则默认为 original（即原版不做 query 数调节）
        self.query_variant = getattr(cfg.MODEL.MASK_FORMER, "QUERY_VARIANT", "original")

        self.model = MaskFormerModel_baseline(cfg)
       
       
        
        # 先DataParallel再载入权重就会有module
        # 载入权重
        if not cfg.inferonly:
            if cfg.MODEL.PRETRAINED_WEIGHTS is not None and os.path.exists(cfg.MODEL.PRETRAINED_WEIGHTS):
                # self.load_model(cfg.MODEL.PRETRAINED_WEIGHTS)
                self.load_state_dict_fix_numpy(cfg.MODEL.PRETRAINED_WEIGHTS)
                print("loaded pretrain mode:{}".format(cfg.MODEL.PRETRAINED_WEIGHTS))
        else:
            if cfg.MODEL.INFER_PRETRAINED_WEIGHTS is not None and os.path.exists(cfg.MODEL.INFER_PRETRAINED_WEIGHTS):
                # self.load_model(cfg.MODEL.INFER_PRETRAINED_WEIGHTS)
                self.load_state_dict_fix_numpy(cfg.MODEL.INFER_PRETRAINED_WEIGHTS)
                print("loaded pretrain mode:{}".format(cfg.MODEL.INFER_PRETRAINED_WEIGHTS))
        if cfg.local_rank != -1:
            torch.cuda.set_device(cfg.local_rank)
            self.device=torch.device("cuda", cfg.local_rank)
            # torch.distributed.init_process_group(backend="nccl", init_method='env://')
        self.model = self.model.to(self.device)
        if cfg.ngpus > 1:
            self.model = self.model.to(self.device) 
            if not cfg.inferonly:
                self.model=nn.DataParallel(self.model)    
            else:
                self.model=nn.DataParallel(self.model)  
        self._training_init(cfg) 

        # ==============================
        # 按类别共享的 query memory bank（baseline 版本）
        # - 在 baseline 里我们只做“更新和维护”，不改动原有前向结构；
        # - 每个类别最多保存 5 个 query，跨 batch 共享；
        # - 更新时使用 Hungarian 匹配得到的 GT 类别标签。
        # ==============================
        # self.classwise_memory = ClasswiseQueryMemoryModule(
        #     C=cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
        #     Q=self.num_queries,
        #     num_classes=self.num_classes,
        #     per_class_max_queries=5,
        # ).to(self.device)
            # self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[cfg.local_rank], output_device=cfg.local_rank, find_unused_parameters=True)             

        run_name = datetime.datetime.now().strftime("swin-%Y-%m-%d-%H-%M")
        # 打印当前 query 策略，便于区分实验版本
        print(f"[MaskFormer_baseline] QUERY_VARIANT = {self.query_variant}")

    def build_optimizer(self):
        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = self.cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                self.cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and self.cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim
            
        optimizer_type = self.cfg.SOLVER.OPTIMIZER
        # =============================
        # 将 backbone 与其余部分拆成不同的 param group
        # - 方便给 backbone 使用更小的学习率
        # - 默认使用 BACKBONE_MULT 作为缩放因子（若未在 cfg 中定义，则使用 0.1）
        # =============================

        # 如果用了 DataParallel，需要拿到真正的 model 实例
        model_for_optim = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        backbone_params = []
        other_params = []
        for name, param in model_for_optim.named_parameters():
            if not param.requires_grad:
                continue
            # 名称里含有 "backbone" 的统一视为 backbone 参数
            if name.startswith("backbone") or ".backbone." in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

        base_lr = self.last_lr
        # 可以在 yaml 里加 cfg.SOLVER.BACKBONE_MULT，例如 0.1；若没有则默认 0.1
        backbone_mult = getattr(self.cfg.SOLVER, "BACKBONE_MULT", 0.1)
        backbone_lr = base_lr * backbone_mult

        param_groups = [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": other_params,  "lr": base_lr},
        ]

        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                param_groups, lr=base_lr, momentum=0.9, weight_decay=0.0001)
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                param_groups, lr=base_lr)
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")

        if not self.cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(self.cfg, optimizer)

        return optimizer

    def load_model(self, pretrain_weights):
        try:
            state_dict = torch.load(pretrain_weights, map_location='cuda:0')
            ckpt_dict = state_dict['model']
        except:
            with open(pretrain_weights, "rb") as f:
                obj = pickle.load(f)
            ckpt_dict = obj.get("model", obj)  # 有的权重直接是 state_dict
       
        dict_new={}
        print(self.model.state_dict().keys())
        
        print(ckpt_dict.keys())
        print('loaded pretrained weights form %s !' % pretrain_weights)
        if self.cfg.bina:
            for k, v in ckpt_dict.items():
                if k.startswith('module.'):
                    # if 'backbone' in k:
                        # k1=k[7:].replace('backbone','backbone1')
                        # k2=k[7:].replace('backbone','backbone2')
                        # dict_new[k1] = v
                        # dict_new[k2] = v
                    # else:
                    dict_new[k[7:]] = v
                    # dict_new[k[7:]] = v
                else:
                    dict_new[k] = v
        else:
            for k, v in ckpt_dict.items():
                if k.startswith('module.'):
                        dict_new[k[7:]] = v
                else:
                    dict_new[k] = v
            
            
        # print(dict_new.keys())
        self.last_lr = 4e-5
        # self.last_lr = 6e-5 # state_dict['lr']
        self.start_epoch = 70 # state_dict['epoch']
        self.model = load_parallal_model(self.model, dict_new)
        
    # 加载plk权重
    def load_state_dict_fix_numpy(self, weight_pth):
        print(f"Loading weights from: {weight_pth}")
        try:
            state_dict = torch.load(weight_pth, map_location='cuda:0')
            state_dict = state_dict['model']
        except:
            with open(weight_pth, "rb") as f:
                obj = pickle.load(f)
            state_dict = obj.get("model", obj) 
        state_dict_temp={}
        for key in state_dict:
            
            # print(key)
            if key.startswith('module') and not key.startswith('module_list'):
                state_dict_temp[key[7:]] = state_dict[key]
            else:
                state_dict_temp[key] = state_dict[key]
            # state_dict_temp[key] = state_dict[key]
        # 如果是 {'model': ...} 结构，取出 model 部分
        if isinstance(state_dict_temp, dict) and 'model' in state_dict_temp:
            state_dict_temp = state_dict_temp['model']
        
        # 清洗权重：转换 numpy 为 tensor
        new_state_dict = OrderedDict()
        for k, v in state_dict_temp.items():
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            new_state_dict[k] = v
        
        model_dict = self.model.state_dict()
        # criterion_dict = self.criterion.state_dict()
        # print(new_state_dict.keys())
        # print(model_dict.keys())
        # print(criterion_dict.keys())
        loaded_keys = []
        skipped_keys = []

        # 手动加载匹配的参数（忽略 shape 不匹配的）
        for k, v in new_state_dict.items():

            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    model_dict[k] = v
                    loaded_keys.append(k)
                else:
                    skipped_keys.append((k, v.shape, model_dict[k].shape))
            
            else:
                skipped_keys.append((k, v.shape, None))
        

        # 打印加载情况
        print(f"\n✅ Load Summary:")
        print(f"- Total model parameters       : {len(self.model.state_dict())}")
        print(f"- Successfully loaded          : {len(loaded_keys)}")
        print(f"- Skipped due to shape mismatch: {len(skipped_keys)}")
        
        if skipped_keys:
            print("\n❗Skipped keys due to mismatch or missing:")
            for k, shape_loaded, shape_model in skipped_keys:
                print(f"  - {k}: loaded {shape_loaded}, model expects {shape_model}")

        self.last_lr = 6e-5
        # self.last_lr = 6e-5 # state_dict['lr']
        self.start_epoch = 0 # state_dict['epoch']
        # 加载更新后的参数
        self.model.load_state_dict(model_dict, strict=False)
        # self.criterion.load_state_dict(criterion_dict, strict=False) #载入criterion权重
        
    def _training_init(self, cfg):
        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        boundary_weight = cfg.MODEL.MASK_FORMER.BOUNDARY_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]
        self.criterion = SetCriterion(
            self.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            device=self.device
        )

        self.summary_writer = create_summary(0, log_dir=cfg.TRAIN.LOG_DIR)
        self.save_folder = os.path.join(cfg.TRAIN.CKPT_DIR,cfg.task,str(cfg.fold))
        self.optim = self.build_optimizer()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode='max', factor=0.9, patience=10)

    def reduce_mean(self, tensor, nprocs):  # 用于平均所有gpu上的运行结果，比如loss
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= nprocs
        return rt


    # ============================================
    # 使用 class-wise memory 对最终层 query 做一次查询融合
    # - 输入: outputs 字典（来自 MaskFormerModel_baseline.forward）
    #   需要包含: "pred_embds" (B,C,Q), "mask_features" (B,C,H,W)
    # - 输出: 直接在 outputs 上覆盖 "pred_logits" / "pred_masks"
    #   作为新的 segmentation 结果，从而让正向真正“用到” memory bank。
    # ============================================
    def _apply_classwise_memory_on_outputs(self, outputs):
        if not hasattr(self, "classwise_memory"):
            return outputs

        pred_embds = outputs.get("pred_embds", None)   # (B, C, Q)
        mask_features = outputs.get("mask_features", None)  # (B, C, H, W)
        if pred_embds is None or mask_features is None:
            return outputs

        # decoder 最后一层的输出经过 LayerNorm 后是 norm_output (Q,B,C)，
        # 在 decoder 里被 rearrange 成 pred_embds: (B,C,Q)。
        # 这里我们把它还原成 decoder_output 形状 (B,Q,C)，作为查询 token。
        tokens_bqc = pred_embds.permute(0, 2, 1).contiguous()  # (B, Q, C)

        # 用当前最终层 query 去查询 class-wise memory bank（只读，不在这里更新）
        fused_bqc = self.classwise_memory.read_memory(tokens_bqc)  # (B, Q, C)

        # 通过 decoder 的分类头 / mask 头重新计算 logits 和 masks
        # 注意 DataParallel 场景下，需要从 module 里取真正的网络
        model_core = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        predictor = model_core.sem_seg_head.predictor
        class_head = predictor.class_embed
        mask_head = predictor.mask_embed

        fused_logits = class_head(fused_bqc)              # (B, Q, num_classes+1)
        fused_mask_embed = mask_head(fused_bqc)           # (B, Q, mask_dim)
        fused_masks = torch.einsum("bqc,bchw->bqhw", fused_mask_embed, mask_features)

        outputs["pred_logits"] = fused_logits
        outputs["pred_masks"] = fused_masks
        return outputs


    # ============================================
    # 基于 Hungarian 匹配结果，为每个 query 构造 GT 类别标签
    # 用于 class-wise memory 的更新；不影响原有 loss 计算
    # ============================================
    def _build_query_class_labels(self, outputs, targets, indice_dict):
        """根据 SetCriterion 返回的 indices / targets，构造 (B,Q) 的类别标签。

        - 对每个 batch 内样本 b：
          - indices_main_layer[b] = (src_idx, matched_labels)
          - 其中 src_idx 是被匹配到 GT 的 query 索引；matched_labels 是对应的 GT 语义类别
        - 未被匹配到的 query 统一置为 -1（在 memory 更新时忽略）。
        """

        if "pred_logits" not in outputs:
            return None

        mask_cls = outputs["pred_logits"]  # (B, Q, C+1)
        B, Q, _ = mask_cls.shape

        device = mask_cls.device
        class_labels = torch.full((B, Q), -1, dtype=torch.long, device=device)

        main_key = "indices_main_layer"
        if main_key not in indice_dict:
            return class_labels

        indices_main = indice_dict[main_key]

        # indices_main 是长度为 B 的 list，每个元素是 (src_idx, labels)
        for b, pair in enumerate(indices_main):
            if pair is None:
                continue
            src_idx, matched_labels = pair
            if src_idx.numel() == 0:
                continue
            # 保证在和当前 device 一致的设备上
            src_idx = src_idx.to(device)
            matched_labels = matched_labels.to(device)
            # 对应 src_idx 位置填入 GT 类别
            class_labels[b, src_idx] = matched_labels

        return class_labels
    




    def train(self, train_paths_temp, test_paths_temp, n_epochs,writer,fold):

        max_score = 0.5
        epoch=0
        if epoch % 2 == 0  :
            version = 0 
        else:
            version = 0
            # version = int((epoch % 64 + 1)/2)
            
        if self.cfg.dataset == 'EndoVis2017':
            train_dataloader, val_dataloader = dataloaders.get_dataloaders(
                train_paths_temp, test_paths_temp, self.cfg,batch_size=self.cfg.batch_size,num_workers=self.cfg.workers,version=version,bina=self.cfg.bina,)
        elif self.cfg.dataset == 'EndoVis2018':
            train_dataloader, val_dataloader = dataloaders.get_dataloaders(
                train_paths_temp, test_paths_temp, self.cfg,batch_size=self.cfg.batch_size,num_workers=self.cfg.workers,version=version,bina=self.cfg.bina,)
        for epoch in range(self.start_epoch + 1, n_epochs):
           
            if not self.cfg.inferonly:
                train_loss = self.train_epoch(train_dataloader, epoch,writer,fold)
            evaluator_score = self.evaluate(val_dataloader,epoch,writer,fold)
           
            self.scheduler.step(evaluator_score)
            # if torch.distributed.get_rank() == 0:
            os.makedirs(self.save_folder, exist_ok=True)
            
            
            if evaluator_score > max_score:
                max_score = evaluator_score
                ckpt_path = os.path.join(self.save_folder, 'mask2former_Epoch{0}_dice{1:.4f}.pth'.format(epoch, max_score))
                save_state = {'model': self.model.state_dict(),
                            'lr': self.optim.param_groups[0]['lr'],
                            'epoch': epoch}
                torch.save(save_state, ckpt_path)
                print('weights {0} saved success!'.format(ckpt_path))
            # torch.distributed.barrier() 
            self.summary_writer.close()




    def train_epoch(self,data_loader, epoch,writer,fold):
        #在每个 epoch 开始时调用 set_epoch() 方法，然后再创建 DataLoader 迭代器，以使 shuffle 操作能够在多个 epoch 中正常工作
        sampler = getattr(data_loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
    
        self.model.train()
        self.criterion.train()
        load_t0 = time.time()
        
      
        dice_score = []
        losses_list_train = []
        loss_ce_list_train = []
        loss_dice_list_train = []
        loss_mask_list_train = []
        per_class_train={}
        end = time.time()
        for i, (data, target,name) in enumerate(data_loader):  
            train_iou={}
            if self.cfg.bina:
                inputs_left=data[0].to(device=self.device, non_blocking=True)#启用异步传输可以提升效率
                inputs_right= data[1].to(device=self.device, non_blocking=True)
                flow_l2r=data[2].to(device=self.device, non_blocking=True)
                target = target.to(device=self.device, non_blocking=True)
                feat_sam = feat_sam.to(device=self.device, non_blocking=True)
                outputs = self.model(inputs_left,inputs_right,feat_sam,flow_l2r)  
                
                
            else:
                data_time = time.time() - end
                inputs= data.to(device=self.device, non_blocking=True)
                target = target.to(device=self.device, non_blocking=True)
                # torch.cuda.synchronize()
                # t0 = time.time()
                outputs = self.model(inputs)

            # 在 decoder 最后一层得到的 query 基础上，用 class-wise memory 做一次查询融合，
            # 得到新的 pred_logits / pred_masks，再用于后续 loss 和评估。
            # outputs = self._apply_classwise_memory_on_outputs(outputs)
            
            # 计算损失 & 获取 Hungarian 匹配结果（indices + labels）
            losses,indice_dict = self.criterion(outputs, target,matcher_outputs=None)
            append_epoch_jsonl("/data/yjh_files/code/Mask2Former-Simplify-master/yjh/indice/train/{}/{}".format(self.cfg.task,fold), epoch, i, name, indice_dict)
            # torch.cuda.synchronize()
            # iter_time = time.time() - t0
            # end = time.time()
            # if i % 10 == 0:
            #     print(f"[i {i}] data_time={data_time:.3f}s, iter_time={iter_time:.3f}s")
            weight_dict = self.criterion.weight_dict
                        
            loss_ce = 0.0
            loss_dice = 0.0
            loss_mask = 0.0
            for k in list(losses.keys()):
                if k in weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                    if '_ce' in k:
                        loss_ce += losses[k]
                    elif '_dice' in k:
                        loss_dice += losses[k]
                    elif '_mask' in k:
                        loss_mask += losses[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            # loss = 2*loss_ce + 0.75*loss_dice + 1*loss_mask
            loss = loss_ce + loss_dice + loss_mask
            self.model.zero_grad()
            self.criterion.zero_grad()
            loss.backward()
            with torch.no_grad():
                losses_list_train.append(loss.item())
                loss_ce_list_train.append(loss_ce.item())
                loss_dice_list_train.append(loss_dice.item())
                loss_mask_list_train.append(loss_mask.item())
            self.optim.step()

            # ==============================================
            # 在一个 batch 完成前向 + 反向 + 参数更新之后，
            # 利用当前的 query 特征和 GT 类别，更新 class-wise memory。
            # ==============================================
            # if hasattr(self, "classwise_memory"):
            #     with torch.no_grad():
            #         # 1) 取出 decoder 输出的 query 特征 (B, C, Q) -> (B, Q, C)
            #         pred_embds = outputs.get("pred_embds", None)
            #         if pred_embds is not None:
            #             tokens_bqc = pred_embds.permute(0, 2, 1).contiguous()

            #             # 2) 基于 Hungarian indices 构造每个 query 的 GT 类别标签 (B, Q)
            #             class_labels_bq = self._build_query_class_labels(outputs, target, indice_dict)

            #             # 3) 更新按类别共享的 memory bank
            #             #    - ignore_label=0：通常 0 是背景类，这里默认不将背景写入 memory
            #             #      如果你希望背景也参与 memory，可把 ignore_label 改为 None。
            #             self.classwise_memory.update_memory(
            #                 tokens_bqc=tokens_bqc,
            #                 class_labels_bq=class_labels_bq,
            #                 ignore_label=0,
            #             )

            elapsed = int(time.time() - load_t0)
            eta = int(elapsed / (i + 1) * (len(data_loader) - (i + 1)))
            curent_lr = self.optim.param_groups[0]['lr']
            with torch.no_grad():
                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]   
                pred_masks = self.semantic_inference(mask_cls_results, mask_pred_results)
                target[target >7] -= 8
                dice_batch_mean =dice_coeff(pred_masks.argmax(1),target)
                dice_score.append(dice_batch_mean.item())   
            progress = f'\r[train] {i + 1}/{len(data_loader)} epoch:{epoch} {elapsed}(s) eta:{eta}(s) loss:{(np.mean(losses_list_train)):.6f} loss_ce:{(np.mean(loss_ce_list_train)):.6f} loss_dice:{(np.mean(loss_dice_list_train)):.6f} loss_mask:{(np.mean(loss_mask_list_train)):.6f}, lr:{curent_lr:.2e},dice:{(np.mean(dice_score)):.6f}\n '
            print(progress, end=' ')
            # with open('/home/gjy/code_yjh/Mask2Former-Simplify-master/log/log_endovis_2018_resnet50_6_SAM.txt', 'a') as f:
            with open('/data/yjh_files/code/Mask2Former-Simplify-master/log_new/log_'+self.cfg.task+'_'+str(fold)+'.txt', 'a') as f:
                f.write(progress)
            sys.stdout.flush()
        writer.add_scalar('train/total_loss', np.mean(losses_list_train), epoch)
        writer.add_scalar('train/dice_loss', np.mean(loss_dice_list_train), epoch)
        writer.add_scalar('train/bce_loss', np.mean(loss_ce_list_train), epoch)
        writer.add_scalar('train/train_dice', np.mean(dice_score), epoch)
            # writer.add_scalar('train/train_challengIoU', train_iou['challengIoU'], epoch)
            # writer.add_scalar('train/train_IoU', train_iou['IoU'], epoch)
            # writer.add_scalar('train/train_mcIoU', train_iou['mcIoU'], epoch)
            # writer.add_scalar('train/train_mIoU', train_iou['mIoU'], epoch)
            # for i in range(len(train_iou['cIoU_per_class'])):
            #     per_class_train['cIoU_per_class_'+str(i+1)]=torch.tensor(train_iou['cIoU_per_class'][i])                

        return loss.item()
    @torch.no_grad()
    def evaluate(self, eval_loader, epoch, writer,fold):
        # eval_loader.sampler.set_epoch(epoch)
        self.model.eval()
        self.criterion.eval()
        # 每次验证开始前重置 class-wise memory，避免跨 epoch / 跨 fold 污染
        # if hasattr(self, "classwise_memory"):
        #     self.classwise_memory.reset_memory()
        dice_score = []
        # 用于收集各进程的损失
        losses_list_val = []
        loss_ce_list_val = []
        loss_dice_list_val = []
        loss_mask_list_val = []
        
        # 用于收集各进程的预测和目标
        preds_list = []
        targets_list = []
        per_class_val={}
        
        val_pred=torch.zeros((len(eval_loader.dataset),128,128)).to(device=self.device)
        val_target=torch.zeros((len(eval_loader.dataset),128,128)).to(device=self.device)
        val_image=torch.zeros((len(eval_loader.dataset),128,128,3)).to(device=self.device)
        
        # val_mask_cls_results=torch.zeros((len(eval_loader.dataset),100,9)).to(device=self.device)
        # val_mask_pred_results=torch.zeros((len(eval_loader.dataset),100,128,128)).to(device=self.device)
        names=[]
 
        with torch.no_grad():
            for i, (data, target,name) in enumerate(eval_loader):
                if self.cfg.bina:
                    inputs_left=data[0].to(device=self.device, non_blocking=True)#启用异步传输可以提升效率
                    inputs_right= data[1].to(device=self.device, non_blocking=True)
                    flow_l2r=data[2].to(device=self.device, non_blocking=True)
                    gt_mask = target.to(device=self.device, non_blocking=True)
                    feat_sam = feat_sam.to(device=self.device, non_blocking=True)
                    outputs = self.model(inputs_left,inputs_right,feat_sam,flow_l2r)  
                    
                    
                else:
                    inputs= data.to(device=self.device, non_blocking=True)
                    gt_mask = target.to(device=self.device, non_blocking=True)
                    # inputs, gt_mask = dataloaders.cutmix_batch(inputs, gt_mask, p=0.4, alpha=1.0)
                    # outputs = self.model(inputs) 
                    outputs = self.model(inputs)

                # 同样在验证阶段，让最终层 query 先查询一次 class-wise memory，
                # 用融合后的 query 生成 pred_logits / pred_masks。
                # outputs = self._apply_classwise_memory_on_outputs(outputs)

                losses,indice_dict = self.criterion(outputs, gt_mask)
                if not self.cfg.inferonly:
                    append_epoch_jsonl("/data/yjh_files/code/Mask2Former-Simplify-master/yjh/indice/test/{}/{}".format(self.cfg.task,fold), epoch, i, name, indice_dict)

                weight_dict = self.criterion.weight_dict
                            
                loss_ce = 0.0
                loss_dice = 0.0
                loss_mask = 0.0
                for k in list(losses.keys()):
                    if k in weight_dict:
                        losses[k] *= self.criterion.weight_dict[k]
                        if '_ce' in k:
                            loss_ce += losses[k]
                        elif '_dice' in k:
                            loss_dice += losses[k]
                        elif '_mask' in k:
                            loss_mask += losses[k]
                    else:
                        # remove this loss if not specified in `weight_dict`
                        losses.pop(k)
                loss = loss_ce + loss_dice + loss_mask
                
                losses_list_val.append(loss.item())
                loss_ce_list_val.append(loss_ce.item())
                loss_dice_list_val.append(loss_dice.item())
                loss_mask_list_val.append(loss_mask.item())

                # ==============================================
                # 验证 / 推理阶段：使用 **预测标签** 更新 class-wise memory
                # - 不使用 GT / 匹配结果，避免数据泄露；
                # - 每个 epoch 进入 evaluate 前已 reset_memory()；
                # - 仅当前验证过程内在线构建 memory。
                # ==============================================
                # if hasattr(self, "classwise_memory"):
                #     # 1) decoder 输出的 query 特征 (B, C, Q) -> (B, Q, C)
                #     pred_embds_eval = outputs.get("pred_embds", None)
                #     if pred_embds_eval is not None:
                #         tokens_bqc_eval = pred_embds_eval.permute(0, 2, 1).contiguous()

                #         # 2) 使用预测的类别作为 pseudo label：argmax over class dim
                #         #    - 输出 logits 维度: (B, Q, num_classes+1)
                #         #    - 最后一类是 no-object（index = num_classes），会在
                #         #      ClasswiseQueryMemoryModule 中因越界而被自动忽略；
                #         #    - ignore_label=0：通常 0 为背景类，这里默认不写入 memory。
                #         pred_logits_eval = outputs["pred_logits"]  # (B, Q, C+1)
                #         pseudo_labels_bq = pred_logits_eval.argmax(dim=-1)  # (B, Q)

                #         self.classwise_memory.update_memory(
                #             tokens_bqc=tokens_bqc_eval,
                #             class_labels_bq=pseudo_labels_bq,
                #             ignore_label=0,
                #         )
                
                # 生成预测
                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                gt_mask[gt_mask>7] -= 8
                instance=True
                if instance:
                    pred_masks_val_list = []
                    
                    for idx in range(len(mask_cls_results)):
                        pic_name = name[idx]
                        
                        mask_cls_instance= mask_cls_results[idx]
                        mask_pred_instance= mask_pred_results[idx]
                        pred_outputs = instance_inference_pure(
                            "all",mask_cls_instance, mask_pred_instance,pic_name, 
                            test_topk_per_image=10, panoptic_on=False,
                            thing_class_ids=None, box_from_mask=False, mask_bin_thresh=0.0
                            )
                    #     pred_masks_val_list.append(batch_topk_instances_to_semantic_overwrite(pic_name,pred_outputs,7,5))
                    # pred_masks_val=torch.stack(pred_masks_val_list,dim=0)
                    # # print(batch_topk_instances_to_semantic_overwrite(pred_outputs,7,7).max())
                    # index=eval_loader.batch_size
                    # val_pred[index*i:index*i+index,:,:] = pred_masks_val
                    # val_target[index*i:index*i+index,:,:] = gt_mask  #取后一帧

                    # names+=name 
                    
                continue
                # else:
                # # # # #         continue
                #     if self.cfg.inferonly:
                #         pred_masks_val = self.semantic_inference(mask_cls_results, mask_pred_results)
                #         # pred_masks_val = self.semantic_inference_topk(mask_cls_results, mask_pred_results,2)
                #     else:
                #         pred_masks_val = self.semantic_inference(mask_cls_results, mask_pred_results)
                #     index=eval_loader.batch_size      
                  

                #     val_pred[index*i:index*i+index,:,:] = pred_masks_val.argmax(dim=1)
                    # if self.cfg.inferonly:
                    #     inputs = F.interpolate(inputs, size=(128, 128), mode="bilinear", align_corners=False)
                    #     x_denorm = denorm_bchw(inputs,  [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]).clamp(0, 1)*255
                    #     val_image[index*i:index*i+index,:,:,:] = x_denorm.permute(0,2,3,1)
                    #     class_ids = [0,1,2,3,4,5,6,7]
                    #     class_names = ["0","1","2","3","4","5","6","7"]
                    #     for i in range(len(mask_cls_results)):
                    #         if name[i] == "seq_15_frame129":
                    #             index=eval_loader.batch_size      
                    #             # pred_masks_val=get_segimg(pred_masks_val)
                    #             pred_masks_val=pred_masks_val
                    #             # val_mask_cls_results[index*i:index*i+index,:,:] = mask_cls_results
                    #             # val_mask_pred_results[index*i:index*i+index,:,:,:] = mask_pred_results 
                    #             fig = plot_query_semseg_distribution(
                    #             gt=gt_mask[i].unsqueeze(0),
                    #             class_ids=class_ids,
                    #             class_names=class_names,
                    #             base=None,   # 没有就传 None
                    #             ours=(mask_cls_results[i].unsqueeze(0), mask_pred_results[0].unsqueeze(0)),   # 没有就传 None
                    #             ignore_index=255,
                    #             mode="log_prob",        # 推荐先用 "prob"（贴近你现在的 semseg），再试 "raw"
                    #             bins=220,
                    #             smooth_sigma=2.0,
                    #             max_pos=60000,
                    #             max_neg=60000,
                    #             seed=0,
                    #             share_x=True,
                    #             title="Comparison of Query-based SemSeg Score Distribution",
                    #             save_path="/data/yjh_files/code/Mask2Former-Simplify-master/yjh/query_semseg_dist_good.png"
                    #         )
                    #             exit()


                val_pred[index*i:index*i+index,:,:] = pred_masks_val.argmax(dim=1)
                val_target[index*i:index*i+index,:,:] = gt_mask 
                names+=name 
            if self.cfg.inferonly:
                # # 计算二分类mask的dice和iou
                iou_mean, dice_mean = compute_iou_and_dice(val_pred, val_target)
                print("IoU mean:", iou_mean.item())
                print("Dice mean:", dice_mean.item())

          
           
            # # =======================
            # # 筛选出 GT 中包含标签 7 的样本名
            # # =======================
            # # 如果你的标签是从 0 开始（0 是背景，1~7 是类别），
            # # 且“label7”指的就是像素值为 7 的类别，则下面直接用 7。
            # # 如果你的第 7 类在 mask 里实际是 6，请把 7 改成 6。
            # LABEL_TO_CHECK = 5

            # # 把 GT mask 移到 CPU，便于统计
            # targets_cpu = targets_all.detach().cpu().long()  # [N, H, W]
            # N, H, W = targets_cpu.shape

            # # 展平成 [N, H*W]，判断每张图是否包含该标签
            # targets_flat = targets_cpu.view(N, -1)
            # has_label = (targets_flat == LABEL_TO_CHECK).any(dim=1)  # [N]

            # print("===== Images whose GT contains label", LABEL_TO_CHECK, "=====")
            # for idx in range(N):
            #     if has_label[idx]:
            #         print(names[idx])
            # print("============================================")
            # exit()
            # # =======================
            # # 统计测试集中各标签出现情况
            # # =======================
            # # 1) 按像素数量：整个测试集中，每个标签的像素总数
            # # 2) 按图像数量：有多少张图像中出现过该标签

            # # 先把所有 GT mask 移到 CPU，便于统计
            # targets_cpu = targets_all.detach().cpu().long()  # [N, H, W]

            # # 仅根据实际出现过的标签来统计，避免对不存在的类别做无意义输出
            # unique_labels = torch.unique(targets_cpu)

            # pixel_count_per_class = {}
            # image_count_per_class = {}

            # # 将 [N, H, W] 展平到 [N, H*W]，便于按图像做 any()
            # N, H, W = targets_cpu.shape
            # targets_flat = targets_cpu.view(N, -1)

            # for c in unique_labels.tolist():
            #     # 所有像素中，该标签的像素总数
            #     class_mask_flat = (targets_flat == c)
            #     pixel_count = class_mask_flat.sum().item()

            #     # 有多少张图像中出现过该标签（至少一个像素）
            #     image_count = class_mask_flat.any(dim=1).sum().item()

            #     pixel_count_per_class[c] = pixel_count
            #     image_count_per_class[c] = image_count

            # print("===== Label statistics on evaluation set (GT masks) =====")
            # for c in sorted(pixel_count_per_class.keys()):
            #     print(f"Label {c}: pixels = {pixel_count_per_class[c]}, images = {image_count_per_class[c]}")
            # print("========================================================")
            
             #  # 计算二分类mask的dice和iou
            # iou_mean, dice_mean = compute_iou_and_dice(val_pred, val_target)
            # print("IoU mean:", iou_mean.item())
            # print("Dice mean:", dice_mean.item())
           
             # 循环结束后：统一设置 preds/targets（后面 eval_endovis 也会用）
                preds_all = val_pred.long()
                targets_all = val_target.long()

              
                num_classes = 7
                iou_thresh = float(getattr(self.cfg, 'EVAL_IOU_THRESH', 0.8))
                eps = 1e-6

                # (1) binary dice
                _, _, dice_bin = self._eval_binary_fg_and_dice(preds_all, targets_all, eps=eps)

                # (2) per-class IoU + TP/TN/FP/FN
                iou_mat, per_class_cm, per_class_auc = self._eval_image_level_per_class_tp_tn_fp_fn(
                    preds_all=preds_all,
                    targets_all=targets_all,
                    num_classes=num_classes,
                    iou_thresh=iou_thresh,
                    eps=eps,
                )

                # AUC macro / micro（可选，但你之前代码里有）
                auc_vals = [v for v in per_class_auc.values() if not (v != v)]
                auc_macro = float(np.mean(auc_vals)) if len(auc_vals) > 0 else float('nan')
                try:
                    gt_present_mat = torch.stack(
                        [(targets_all == cid).flatten(1).sum(1) > 0 for cid in range(1, num_classes + 1)],
                        dim=1,
                    )
                    y_true_micro = gt_present_mat.to(torch.int64).flatten().cpu().numpy()
                    y_score_micro = iou_mat.detach().flatten().cpu().numpy()
                    auc_micro = roc_auc_score(y_true_micro, y_score_micro)
                except Exception:
                    auc_micro = float('nan')

                # TensorBoard：每类 AUC
                for cid in range(1, num_classes + 1):
                    auc_c = per_class_auc[cid]
                    if not (auc_c != auc_c):
                        writer.add_scalar(f'val/class_{cid}_auc', auc_c, epoch)

                save_dir = os.path.join(self.save_folder, 'eval_metrics')
                os.makedirs(save_dir, exist_ok=True)

                # (a) 混淆矩阵图：每类一个 2x2 子图
                ncols = 4
                nrows = int(math.ceil(num_classes / ncols))
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.5*ncols, 3.0*nrows), dpi=200)
                axes = np.array(axes).reshape(-1)
                for idx, cid in enumerate(range(1, num_classes + 1)):
                    ax = axes[idx]
                    disp = ConfusionMatrixDisplay(per_class_cm[cid], display_labels=['neg', 'pos'])
                    disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')
                    ax.set_title(f'class {cid} (thr={iou_thresh:.2f})')
                for j in range(num_classes, len(axes)):
                    axes[j].axis('off')
                fig.tight_layout()
                cm_path = os.path.join(save_dir, f'confusion_matrix_per_class_epoch_{epoch:04d}.png')
                fig.savefig(cm_path)
                plt.close(fig)

                # (b) AUC 柱状图
                fig, ax = plt.subplots(figsize=(8, 3), dpi=200)
                xs = np.arange(1, num_classes + 1)
                ys = [per_class_auc[c] if not (per_class_auc[c] != per_class_auc[c]) else 0.0 for c in xs]
                ax.bar(xs, ys)
                ax.set_xticks(xs)
                ax.set_xlabel('class id')
                ax.set_ylabel('AUC')
                ax.set_ylim(0.0, 1.0)
                ax.set_title(f'Per-class AUC (macro={auc_macro:.4f}, micro={auc_micro:.4f}) epoch={epoch}')
                fig.tight_layout()
                auc_fig_path = os.path.join(save_dir, f'auc_per_class_epoch_{epoch:04d}.png')
                fig.savefig(auc_fig_path)
                plt.close(fig)

                print(f'[eval-metrics] binary_dice={dice_bin:.4f} auc_macro={auc_macro:.4f} auc_micro={auc_micro:.4f} cm={cm_path} auc_fig={auc_fig_path}')
                writer.add_scalar('val/binary_dice', dice_bin, epoch)
                if not (auc_macro != auc_macro):
                    writer.add_scalar('val/class_auc_macro', auc_macro, epoch)
                if not (auc_micro != auc_micro):
                    writer.add_scalar('val/class_auc_micro', auc_micro, epoch)
            dice_all=[]
            # val_iou = eval_endovis(preds_all, targets_all, num_classes=7, ignore_background=True, device='cpu')
            val_iou = eval_endovis(preds_all, targets_all, num_classes=7, ignore_background=True, device=self.device)
            if self.cfg.inferonly:
                for i in range(len(names)):
                    dice_mean_new=dice_coeff(preds_all[i,:,:].unsqueeze(0), targets_all[i,:,:].unsqueeze(0).to(self.device))
                    dice_all.append(dice_mean_new.detach().cpu().numpy())
                    # visanddraw(preds_all[i,:,:], targets_all[i,:,:].to(self.device),names[i],dice_mean_new,'endovis_2017_baseline_swins_20_0_3/',val_image[i])
                    overlay(preds_all[i,:,:], targets_all[i,:,:].to(self.device),names[i],dice_mean_new,'endovis_2017_baseline_swins_20_0_3/',val_image[i])
                dice=np.mean(dice_all)
            # dice_all = dice_coeff(preds_all, targets_all)
            # score = dice_all.mean().item()
            score=val_iou['IoU'] #就是ISINET IOU，对mask求平均
            print(val_iou)
            print(np.mean(losses_list_val))
            # print('IOU:',val_iou['IoU'])
            # 记录日志
            # exit()
            writer.add_scalar('val/val_loss', np.mean(losses_list_val), epoch)
            writer.add_scalar('val/val_dice_loss', np.mean(loss_dice_list_val), epoch)
            writer.add_scalar('val/val_ce_loss', np.mean(loss_ce_list_val), epoch)
            writer.add_scalar('val/val_challengIoU', val_iou['challengIoU'], epoch)
            writer.add_scalar('val/val_IoU', val_iou['IoU'], epoch)  #对计算iou后每个mask平均
            writer.add_scalar('val/val_mcIoU', val_iou['mcIoU'], epoch)
            writer.add_scalar('val/val_mIoU', val_iou['mIoU'], epoch) #对每张图像平均（无视图像中mask数量的差异，每帧图像的贡献平等，无论其包含的像素数量或类别数量多少）
            for i in range(len(val_iou['cIoU_per_class'])):
                per_class_val['cIoU_per_class_'+str(i+1)]=torch.tensor(val_iou['cIoU_per_class'][i])
            writer.add_scalars('val/val_cIoU_per_class', per_class_val, epoch)
            # writer.add_scalar('val/val_dice', score, epoch)
            # print('val evaluate dice: {0}, IoU: {1},mIoU: {2}/n'.format(score))
            print('evaluate dice: {0},loses: {1},/n'.format(score,np.mean(losses_list_val)))
            with open('/data/yjh_files/code/Mask2Former-Simplify-master/log_new/log_'+self.cfg.task+'_'+str(fold)+'.txt', 'a') as f: 
                f.write('evaluate IIoU: {0},losses:{1}/n'.format(score,np.mean(losses_list_val)))
                progress = f'loss:{(np.mean(losses_list_val)):.6f} loss_ce:{(np.mean(loss_ce_list_val)):.6f} loss_dice:{(np.mean(loss_dice_list_val)):.6f} loss_mask:{(np.mean(loss_mask_list_val)):.6f}\n '
                f.write(progress) 
            del val_pred,val_target
        return score

   
    def _get_dice(self, predict, target):    
        smooth = 1e-5 
        # print(predict.max(),predict.min())   
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1)
        den = predict.sum(-1) + target.sum(-1) 
        score = (2 * num + smooth).sum(-1) / (den + smooth).sum(-1)
        return score.mean()
    
    def _get_binary_mask(self, target):
        # 返回每类的binary mask
        num,y, x = target.size()
        target_onehot = torch.zeros(num,self.num_classes + 1, y, x).to(target.device)
        target_onehot = target_onehot.scatter(dim=1, index=target.unsqueeze(1), value=1) #此处需要改成1
        return target_onehot
    
    def _get_binary_mask_one(self, target):
        
        y, x = target.size()
        target_onehot = torch.zeros(self.num_classes + 1, y, x).to(target.device)
        target_onehot = target_onehot.scatter(dim=0, index=target.unsqueeze(0), value=1)
        return target_onehot[1:-1]


    # =====================================================
    # Eval helpers (inferonly)
    # - 二值前景掩膜(binary)与 Dice
    # - 图像级：按类(类 vs 背景)计算 IoU，并按你定义的规则判定 TP/TN/FP/FN
    # =====================================================
    def _eval_binary_fg_and_dice(self, preds_all, targets_all, eps: float = 1e-6):
        """二值前景(>0) vs 背景(=0) 的 Dice。

        preds_all/targets_all: (N,H,W) 取值为 {0..K}
        """
        preds_all = preds_all.long()
        targets_all = targets_all.long()

        pred_fg = (preds_all > 0)
        gt_fg = (targets_all > 0)

        inter = (pred_fg & gt_fg).flatten(1).sum(1).float()
        denom = pred_fg.flatten(1).sum(1).float() + gt_fg.flatten(1).sum(1).float()
        dice_bin = ((2.0 * inter + eps) / (denom + eps)).mean().item()
        return pred_fg, gt_fg, dice_bin

    def _eval_image_level_per_class_tp_tn_fp_fn(
        self,
        preds_all,
        targets_all,
        num_classes: int,
        iou_thresh: float,
        eps: float = 1e-6,
    ):
        """按类(1..K)做图像级判定。

        规则(每张图、每个类 cid):
        - 计算该类与背景对应的二分类 IoU (pred==cid vs gt==cid)
        - 若 IoU > thresh -> TP
        - 若 GT 和 Pred 都没有该类(即 union==0) -> TN
        - 若 IoU <= thresh:
            - pred_only(非重合预测面积) > gt_only(非重合GT面积) -> FP
            - 否则 -> FN

        返回:
        - iou_mat: (N, K)
        - per_class_cm: dict[cid] = 2x2 ndarray [[TN,FP],[FN,TP]]
        - per_class_auc: dict[cid] = AUC (用 IoU 作为 score，y_true=GT是否出现该类)
        """
        preds_all = preds_all.long()
        targets_all = targets_all.long()
        device = preds_all.device

        N = preds_all.shape[0]
        iou_mat = torch.zeros((N, num_classes), device=device, dtype=torch.float32)

        per_class_cm = {}
        per_class_auc = {}

        for cid in range(1, num_classes + 1):
            p = (preds_all == cid)
            g = (targets_all == cid)

            inter = (p & g).flatten(1).sum(1).float()
            union = (p | g).flatten(1).sum(1).float()
            iou = torch.where(union > 0, inter / (union + eps), torch.zeros_like(union))
            iou_mat[:, cid - 1] = iou

            # 面积
            pred_only = (p & (~g)).flatten(1).sum(1).float()
            gt_only = (g & (~p)).flatten(1).sum(1).float()

            # 你的 TP/TN/FP/FN 规则
            tp = (union > 0) & (iou > iou_thresh)
            tn = (union == 0)
            undecided = (~tp) & (~tn)  # union>0 & iou<=thr
            fp = undecided & (pred_only > gt_only)
            fn = undecided & (~(pred_only > gt_only))  # pred_only <= gt_only

            # 组 2x2 confusion matrix
            tn_c = int(tn.sum().item())
            fp_c = int(fp.sum().item())
            fn_c = int(fn.sum().item())
            tp_c = int(tp.sum().item())
            per_class_cm[cid] = np.array([[tn_c, fp_c], [fn_c, tp_c]], dtype=np.int64)

            # AUC：y_true=GT是否出现该类；y_score=IoU
            y_true = (g.flatten(1).sum(1) > 0).to(torch.int64).detach().cpu().numpy()
            y_score = iou.detach().cpu().numpy()
            auc_c = float('nan')
            try:
                if len(set(y_true.tolist())) == 2:
                    auc_c = roc_auc_score(y_true, y_score)
            except Exception as e:
                print(f'[eval-metrics] class {cid} roc_auc_score failed: {e}')
            per_class_auc[cid] = auc_c

        return iou_mat, per_class_cm, per_class_auc


    def semantic_inference(self, mask_cls, mask_pred):       
        mask_cls = F.softmax(mask_cls, dim=-1)[...,:-1] #(batchsize, num_queries, num_classes) ,值表示每个类别的概率
        # semseg= F.softmax(semseg, dim=1)
        # print( mask_cls[0])#此处对0-7类别（考虑背景）做softmax，之后才可以用argmax得到类别
        mask_pred = mask_pred.sigmoid()   #(batchsize, num_queries, h, w) ,值表示每个像素属于每个mask的概率(即不属于背景的概率)
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred) 
         # (batchsize, num_classes, h, w)
        return semseg
    
    def semantic_inference_topk(self, mask_cls, mask_pred,topk=2):  
      
          # 初始处理
        mask_cls = F.softmax(mask_cls, dim=-1)[..., 1:-1]# (B, Q, C)
        mask_pred = mask_pred.sigmoid() 
        mask_pred=nn.Threshold(0.5,0)(mask_pred)# (B, Q, H, W)
        B, Q, H, W = mask_pred.shape
        C = mask_cls.shape[-1]

        # Step 1: 创建稀疏矩阵（保留每个Q的最大类别）
        max_values, max_indices = torch.max(mask_cls, dim=2, keepdim=True)
        mask_cls_sparse = torch.zeros_like(mask_cls)
        mask_cls_sparse.scatter_(2, max_indices, max_values)  # (B, Q, C)

        # Step 2: 转置维度并获取topk
        mask_cls_transposed = mask_cls_sparse.permute(0, 2, 1)  # (B, C, Q)
        topk_values, topk_indices = torch.topk(mask_cls_transposed, k=topk, dim=2)  # (B, C, topk)

        # Step 3: 创建有效掩码
        valid_mask = topk_values > 0  # (B, C, topk)
        
        mask_pred_all=[]
        semseg_all=[]
        for b in range(B):
            # print(valid_mask[b])
            topk_index=[topk_indices[b].flatten()[i] for i in range(len(topk_indices[b].flatten())) if valid_mask[b].flatten()[i]]
            topk_index=torch.tensor(topk_index)
            topk_indices_batch=torch.unique(topk_index).to(mask_cls.device)
            # topk_indices_batch=torch.unique(topk_indices[b])
            mask_pred_batch= torch.index_select(mask_pred[b], 0, topk_indices_batch)
            mask_cls_batch= torch.index_select(mask_cls[b], 0, topk_indices_batch)
            # mask_pred_all.append(mask_pred_batch)
            # mask_cls_all.append(mask_cls_batch)
            semseg_batch = torch.einsum("qc,qhw->chw", mask_cls_batch, mask_pred_batch)
            semseg_all.append(semseg_batch)
        semseg=torch.stack(semseg_all)
        return semseg

  
if __name__ == '__main__':
    def _get_binary_mask( num_classes,target):
        # 返回每类的binary mask
        num,y, x = target.size()
        target_onehot = torch.zeros(num,num_classes + 1, y, x).to(target.device)
        target_onehot = target_onehot.scatter(dim=1, index=target.unsqueeze(1), value=1) #index表示在该维度上进行拓展
        return target_onehot
    
    target = torch.tensor([
    [[0, 1], 
     [2, 0]],
    
    [[1, 2], 
     [0, 1]]
], dtype=torch.long) 
    num_classes = 2 
    output=_get_binary_mask(num_classes, target)
    print(output)