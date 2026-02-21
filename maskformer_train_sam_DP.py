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
from modeling.MaskFormerModel import MaskFormerModel
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
from yjh.utils_yjh import instance_inference_pure,batch_topk_instances_to_semantic_overwrite,visanddraw,semantic_inference_with_bg

import torch
import torch.nn as nn
# from transformers import get_cosine_schedule_with_warmup
import torch
import torch.nn as nn
from typing import Iterable, Set, Dict, List, Tuple
import math
from torch.optim.lr_scheduler import _LRScheduler


def vis(pred_mask,name):
        palette = np.array([
        [0, 0, 0],        # class 0 - black
        [255, 0, 0],      # class 1 - red
        [0, 255, 0],      # class 2 - green
        [0, 0, 255],      # class 3 - blue
        [255, 255, 0],    # class 4 - yellow
        [255, 0, 255],    # class 5 - magenta
        [0, 255, 255],    # class 6 - cyan
        [255, 255, 255]   # class 7 - white
    ], dtype=np.uint8)
        #处理第一个样本
        z_n1 = pred_mask.detach().cpu().numpy().astype(np.uint8)
        
        # 检查数值范围 (重要!)
        z_n1 = np.clip(z_n1, 0, len(palette)-1)

        color_mask1 = palette[z_n1]  # 自动广播到 (H, W, 3)
        
        cv2.imwrite(
            f'/home/yjh/code_yjh_bishe/Mask2Former-Simplify-master/test4/pred/{name}.png',
            cv2.cvtColor(color_mask1, cv2.COLOR_RGB2BGR)
        )

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

def dice_coeff(pred, target, num_cls=7, epsilon=1e-6): #hard dice 更贴近实际，在计算loss时一般采用soft dice
 
    """
    pred: (batch_size, num_cls, H, W)  # 模型输出,不含背景类(类别1~7)
    target: (batch_size, H, W)         # 真实标签,包含背景0类(0~7)
    num_cls: 非背景类别数(此处为7)
    """
    # 确保 target 的类别范围合法
    assert target.max() <= num_cls, "Target 包含超出模型预测能力的类别"
    # get_label(pred,target)
   
    # pred_class = torch.argmax(pred, dim=1)          # (batch, H, W) 值为 0~6
                                   # 对齐到 target 的类别（1~7）
    pred_class = pred.long()
    pred_one_hot = F.one_hot(pred_class, num_classes=num_cls+1)  # (batch, H, W, num_cls+1)
    pred_one_hot = pred_one_hot.permute(0, 3, 1, 2).float()[:,:,:,:]      # (batch, num_cls+1, H, W)
    # (batch, num_cls+1, H, W)
    # pred_one_hot = pred
    # 将 target 转为 one-hot（含背景类0）
    target_one_hot = F.one_hot(target.long(), num_classes=num_cls+1)  # (batch, H, W, num_cls+1)
    target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()[:,:,:,:]      # (batch, num_cls+1, H, W)
    # 计算各维度求和
    intersection = torch.sum(pred_one_hot * target_one_hot, dim=(0, 2, 3))  # shape (num_cls+1,)
    union = torch.sum(pred_one_hot, dim=(0, 2, 3)) + torch.sum(target_one_hot, dim=(0, 2, 3))

    # 计算 Dice（排除背景0类）
    dice_scores = (2.0 * intersection + epsilon) / (union + epsilon)
    return dice_scores[1:].mean()  # 只取类别1~7的平均值


# warmup策略
class WarmupPolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, warmup_factor=1.0/3, warmup_iters=500, last_epoch=-1):
        self.max_iters = max_iters
        self.power = power
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        super(WarmupPolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_iters:
            # linear warmup
            alpha = step / float(self.warmup_iters)
            factor = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            # poly decay
            factor = (1 - step / self.max_iters) ** self.power
            return [base_lr * factor for base_lr in self.base_lrs]

class MaskFormer():
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_queries = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        self.size_divisibility = cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY
        self.num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        # self.device = torch.device("cuda", cfg.LOCAL_RANK)
        self.is_training = cfg.MODEL.IS_TRAINING
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.last_lr = cfg.SOLVER.LR
        self.start_epoch = 0
        self.model = MaskFormerModel(cfg)
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
                self.model=nn.DataParallel(self.model,device_ids=[0,1,2,3])    
            else:
                self.model=nn.DataParallel(self.model,device_ids=[0,1])   
            # self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[cfg.local_rank], output_device=cfg.local_rank, find_unused_parameters=True)             

        self._training_init(cfg)

        run_name = datetime.datetime.now().strftime("swin-%Y-%m-%d-%H-%M")

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
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                self.model.parameters(), self.last_lr, momentum=0.9, weight_decay=0.0001)
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                self.model.parameters(), self.last_lr)
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")

        if not self.cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(self.cfg, optimizer)

        return optimizer
    



        
    def _build_optimizer_with_pretrained_groups(self,
                       # 预训练权重中成功加载的参数名集合
        base_lr: float = 8e-5,
        lr_mult_pretrained: float = 0.5,        # 预训练层学习率倍率（预训练层用更小lr）
        weight_decay: float = 0.01,
        use_adamw: bool = True,
        extra_no_weight_decay_names: Iterable[str] = (),  # 额外按名字精确匹配禁用WD
    ) -> torch.optim.Optimizer:
        """
        根据 “是否来自预训练” × “是否 no_weight_decay” 划分参数组：
        - 预训练+decay
        - 预训练+no_decay
        - 新层+decay
        - 新层+no_decay
        """

        # 这些模块的 weight 通常不做 weight decay（规范做法）
        no_decay_modules = (
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm,
            nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d
        )

        # 建立 “模块前缀 -> 模块对象” 映射，便于从参数名定位其父模块
        # 例如 param name: "backbone.stem.conv1.weight"
        #   -> module_path: "backbone.stem.conv1"
        module_map: Dict[str, nn.Module] = dict(self.model.named_modules())

        decay_pre, no_decay_pre = [], []
        decay_new, no_decay_new = [], []

        for full_name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "module" in full_name:
                full_name = full_name.replace("module.", "")
            # 1) 判断该参数是否属于预训练命中集合
            is_pretrained = full_name in self.loaded_keys

            # 2) 解析所属模块 & 子名
            if "." in full_name:
                module_path, child_name = full_name.rsplit(".", 1)
                module = module_map.get(module_path, None)
            else:
                module_path, child_name = "", full_name
                module = self.model  # 参数直接挂在顶层模型上

            # 3) 是否 no_weight_decay
            #   规则：
            #   - 所属模块是规范的norm模块（例如 BN/LN/GN 等）的 weight/bias
            #   - bias 参数
            #   - 在 extra_no_weight_decay_names 中显式指定的参数
            no_wd = False
            if child_name == "bias":
                no_wd = True
            if module is not None and isinstance(module, no_decay_modules):
                no_wd = True
            if full_name in set(extra_no_weight_decay_names):
                no_wd = True

            # 4) 放入对应参数组
            if is_pretrained:
                (no_decay_pre if no_wd else decay_pre).append(param)
            else:
                (no_decay_new if no_wd else decay_new).append(param)

        param_groups: List[Dict] = []
        if decay_pre:
            param_groups.append({"params": decay_pre, "lr": base_lr * lr_mult_pretrained, "weight_decay": weight_decay})
        if no_decay_pre:
            param_groups.append({"params": no_decay_pre, "lr": base_lr * lr_mult_pretrained, "weight_decay": 0.0})
        if decay_new:
            param_groups.append({"params": decay_new, "lr": base_lr, "weight_decay": weight_decay})
        if no_decay_new:
            param_groups.append({"params": no_decay_new, "lr": base_lr, "weight_decay": 0.0})

        Optim = torch.optim.AdamW if use_adamw else torch.optim.Adam
        optimizer = Optim(param_groups, lr=base_lr, betas=(0.9, 0.999))
 
        # 可选：打印各组规模，便于核对
        print(
            f"[optimizer] pre(decay/no)={len(decay_pre)}/{len(no_decay_pre)}, "
            f"new(decay/no)={len(decay_new)}/{len(no_decay_new)}, "
            f"groups={len(param_groups)}"
        )
        return optimizer
    
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
            if key.startswith('module') and not key.startswith('module_list'):
                state_dict_temp[key[7:]] = state_dict[key]
            else:
                state_dict_temp[key] = state_dict[key]
        
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
        self.loaded_keys = []
        skipped_keys = []

        # 手动加载匹配的参数（忽略 shape 不匹配的）
        for k, v in new_state_dict.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    model_dict[k] = v
                    self.loaded_keys.append(k)
                else:
                    skipped_keys.append((k, v.shape, model_dict[k].shape))
            else:
                skipped_keys.append((k, v.shape, None))

        

        # 打印加载情况
        print(f"\n✅ Load Summary:")
        print(f"- Total model parameters       : {len(self.model.state_dict())}")
        print(f"- Successfully loaded          : {len(self.loaded_keys)}")
        print(f"- Skipped due to shape mismatch: {len(skipped_keys)}")

        if skipped_keys:
            print("\n❗Skipped keys due to mismatch or missing:")
            for k, shape_loaded, shape_model in skipped_keys:
                print(f"  - {k}: loaded {shape_loaded}, model expects {shape_model}")

        self.last_lr = 8e-5
        # self.last_lr = 6e-5 # state_dict['lr']
        self.start_epoch = 0# state_dict['epoch']
        # 加载更新后的参数
        self.model.load_state_dict(model_dict, strict=False)
        
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
        self.save_folder = os.path.join(cfg.TRAIN.CKPT_DIR,cfg.task)
        # self.optim = self._build_optimizer_with_pretrained_groups()
        self.optim = self.build_optimizer()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode='max', factor=0.9, patience=10)

    def reduce_mean(self, tensor, nprocs):  # 用于平均所有gpu上的运行结果，比如loss
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= nprocs
        return rt
    




    def train(self, train_paths_temp, test_paths_temp, n_epochs,writer):
        

        TrainPath=[]
        ValPath=[]
        # # for i in range(0,9):
        # #     index=0
        # #     for k in train_paths_temp:
                
                
        # #         if 'dataset_'+str(i) in k:
        # #             index+=1
        # #             if index< 169 :
        # #                 TrainPath.append(k)
        # #             else:
        # #                  ValPath.append(k)   
        if self.cfg.dataset == 'EndoVis2017':
            for i in train_paths_temp:
                if 'dataset_6' in i or 'dataset_7' in i:
                    ValPath.append(i)
                else:
                    TrainPath.append(i)
        
        # shuffled_list = copy.deepcopy(train_paths_temp)
        # random.seed(42)
        # random.shuffle(shuffled_list)

        # train = shuffled_list[:1350]

        max_iters = n_epochs * round(len(train_paths_temp)/self.cfg.batch_size)
        # self.scheduler = WarmupPolyLR(
        #     self.optim,
        #     max_iters=max_iters,
        #     power=0.9,
        #     warmup_factor=1.0/3,
        #     warmup_iters=1500  # 依赖 batch_size 和数据集大小
        # )
        
        # test = shuffled_list[1350:]
        max_score = 0.5
        for epoch in range(self.start_epoch + 1, n_epochs):
            if epoch % 2 == 0  :
                version = 0 
            else:
                version = int((epoch % 64+ 1)/2)
                # version = 0
            if self.cfg.dataset == 'EndoVis2017':
                train_dataloader, val_dataloader = dataloaders.get_dataloaders(
                    TrainPath,ValPath, self.cfg,batch_size=self.cfg.batch_size,num_workers=self.cfg.workers,version=version,bina=self.cfg.bina,)
            elif self.cfg.dataset == 'EndoVis2018':
                train_dataloader, val_dataloader = dataloaders.get_dataloaders(
                    train_paths_temp, test_paths_temp, self.cfg,batch_size=self.cfg.batch_size,num_workers=self.cfg.workers,version=version,bina=self.cfg.bina,)
            if not self.cfg.inferonly:
                a=len(train_dataloader)
                train_loss = self.train_epoch(train_dataloader, epoch,writer)
            evaluator_score = self.evaluate(val_dataloader,epoch,writer)
           
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




    def train_epoch(self,data_loader, epoch,writer):
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
        for i, (data, target,feat_sam,name) in enumerate(data_loader):  
                               
            # inputs = batch['images'].to(device=self.device, non_blocking=True)
            # targets = batch['masks']
            # print(data.max())
            train_iou={}
            if self.cfg.bina:
                inputs_left=data[0].to(device=self.device, non_blocking=True)#启用异步传输可以提升效率
                inputs_right= data[1].to(device=self.device, non_blocking=True)
                flow_l2r=data[2].to(device=self.device, non_blocking=True)
                target = target.to(device=self.device, non_blocking=True)
                feat_sam = feat_sam.to(device=self.device, non_blocking=True)
                
                # train_pred=torch.zeros((len(data_loader)*4,128,128))
                # train_target=torch.zeros((len(data_loader)*4,128,128))
                
                

                outputs = self.model(inputs_left,inputs_right,feat_sam,flow_l2r)  
                
                
            else:
                data_time = time.time() - end
                inputs= data.to(device=self.device, non_blocking=True)
                target = target.to(device=self.device, non_blocking=True)
                torch.cuda.synchronize()
                t0 = time.time()
                feat_sam = feat_sam.to(device=self.device, non_blocking=True)                                              
                outputs = self.model(inputs,feat_sam)  
               
                          
            losses = self.criterion(outputs, target)
            torch.cuda.synchronize()
            iter_time = time.time() - t0
            end = time.time()
            if i % 10 == 0:
                print(f"[i {i}] data_time={data_time:.3f}s, iter_time={iter_time:.3f}s")
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
                
            # with torch.no_grad():
            #     losses_list_train.append(reduce_value(loss).item())
            #     loss_ce_list_train.append(reduce_value(loss_ce).item())
            #     loss_dice_list_train.append(reduce_value(loss_dice).item())
            #     loss_mask_list_train.append(reduce_value(loss_mask).item())
                
               
            # loss = self.reduce_mean(loss, dist.get_world_size())
            self.optim.step()
            # self.scheduler.step()

            elapsed = int(time.time() - load_t0)
            eta = int(elapsed / (i + 1) * (len(data_loader) - (i + 1)))
            curent_lr = self.optim.param_groups[0]['lr']
  
           
            # progress = f'\r[train] {i + 1}/{len(data_loader)} epoch:{epoch} {elapsed}(s) eta:{eta}(s) loss:{(np.mean(losses_list_train)):.6f} loss_ce:{(np.mean(loss_ce_list_train)):.6f} loss_dice:{(np.mean(loss_dice_list_train)):.6f} loss_mask:{(np.mean(loss_mask_list_train)):.6f}, lr:{curent_lr:.2e} '
            # # progress = f'\r[train] {i + 1}/{len(data_loader)} epoch:{epoch} {elapsed}(s) eta:{eta}(s) loss:{(np.mean(losses_list)):.6f} loss_ce:{(np.mean(loss_ce_list)):.6f} loss_dice:{(np.mean(loss_dice_list)):.6f}, lr:{curent_lr:.2e}  '
            # print(progress, end=' ')
           
            
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]            
            pred_masks = self.semantic_inference(mask_cls_results, mask_pred_results)  
            batch_size = target.shape[0]
            dice_scores = []
            for j in range(batch_size):
                gt_binary_mask = self._get_binary_mask_one(target[j])
                dice = self._get_dice(pred_masks[j], gt_binary_mask.to(self.device))
                dice_scores.append(dice)
            batch_dice = torch.mean(torch.stack(dice_scores))
            # dice_score.append(reduce_value(batch_dice, average=True).item())  
            dice_score.append(batch_dice.item())                  
            
            # gt_mask =target[0]
            # gt_binary_mask = self._get_binary_mask_one(gt_mask)
            # # dice=dice_coeff(pred_masks, gt_binary_mask.to(self.device),num_cls=7,epsilon=1e-6) 
            # dice = self._get_dice(pred_masks[0], gt_binary_mask.to(self.device))
            
            # dice_score.append(reduce_value(dice).item())
            # print('dice:{}:'.format(np.mean(dice_score)))
            
        
        
            #16*64*128*12      
            # mask_cls_results_train = outputs["pred_logits"]
            # mask_pred_results_train = outputs["pred_masks"]
            # pred_masks_train = self.semantic_inference(mask_cls_results_train, mask_pred_results_train)          
            # losses = self.criterion(outputs, target)
            # weight_dict = self.criterion.weight_dict
            # # 每个epoch只取第一张图计算iou
            # pred_mask_train = pred_masks_train[:]
            # gt_mask_train=target[:]
            
            # # pred_mask_train = pred_masks_train[0].unsqueeze(0)
            # # gt_mask_train=target[0].unsqueeze(0)
            # # gt_binary_mask_train = self._get_binary_mask(gt_mask_train)
            # dice=dice_coeff(pred_mask_train, gt_mask_train,num_cls=7,epsilon=1e-6) #此处应该是8而不是7
            # # gt_binary_mask_train = self._get_binary_mask_one(gt_mask_train)
            # # dice = self._get_dice(pred_mask_train, gt_binary_mask_train.to(self.device))
            # dice_score.append(dice.item())
            # # train_pred[i, :, :]=pred_mask_train.argmax(dim=0)
            # # train_target[i, :, :]=gt_mask_train          
            # loss_ce = 0.0
            # loss_dice = 0.0
            # loss_mask = 0.0
            # for k in list(losses.keys()):
            #     if k in weight_dict:
            #         losses[k] *= self.criterion.weight_dict[k]
            #         if '_ce' in k:
            #             loss_ce += losses[k]
            #         elif '_dice' in k:
            #             loss_dice += losses[k]
            #         elif '_mask' in k:
            #             loss_mask += losses[k]
            #     else:
            #         # remove this loss if not specified in `weight_dict`
            #         losses.pop(k)
            # loss = loss_ce + loss_dice + loss_mask
            # with torch.no_grad():
            #     losses_list_train.append(loss.item())
            #     loss_ce_list_train.append(loss_ce.item())
            #     loss_dice_list_train.append(loss_dice.item())
            #     loss_mask_list_train.append(loss_mask.item())

            # self.model.zero_grad()
            # self.criterion.zero_grad()
            # loss.backward()
            
            # # os.environ['MASTER_ADDR'] = 'localhost'
            # # os.environ['MASTER_PORT'] = '12345'
            # # dist.init_process_group(backend='nccl',rank=0, world_size = 1)

            # # loss = self.reduce_mean(loss, dist.get_world_size())
            # self.optim.step()  
            # if dist.get_rank() == 0:
            progress = f'\r[train] {i + 1}/{len(data_loader)} epoch:{epoch} {elapsed}(s) eta:{eta}(s) loss:{(np.mean(losses_list_train)):.6f} loss_ce:{(np.mean(loss_ce_list_train)):.6f} loss_dice:{(np.mean(loss_dice_list_train)):.6f} loss_mask:{(np.mean(loss_mask_list_train)):.6f}, lr:{curent_lr:.2e},dice:{(np.mean(dice_score)):.6f}\n '
            print(progress, end=' ')
            # with open('/home/gjy/code_yjh/Mask2Former-Simplify-master/log/log_endovis_2018_resnet50_6_SAM.txt', 'a') as f:
            with open('/home/yjh/code_yjh_bishe/Mask2Former-Simplify-master/log_new/log_'+self.cfg.task+'.txt', 'a') as f:
                f.write(progress)
            sys.stdout.flush()
            

        # if dist.get_rank() == 0:
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
    def evaluate(self, eval_loader, epoch, writer):
        # eval_loader.sampler.set_epoch(epoch)
        self.model.eval()
        self.criterion.eval()
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
        names=[]
 
        with torch.no_grad():
            for i, (data, target,feat_sam,name) in enumerate(eval_loader):
                
                if self.cfg.bina:
                    inputs_left=data[0].to(device=self.device, non_blocking=True)#启用异步传输可以提升效率
                    inputs_right= data[1].to(device=self.device, non_blocking=True)
                    flow_l2r=data[2].to(device=self.device, non_blocking=True)
                    gt_mask = target.to(device=self.device, non_blocking=True)
                    feat_sam = feat_sam.to(device=self.device, non_blocking=True)

                    # train_pred=torch.zeros((len(data_loader)*4,128,128))
                    # train_target=torch.zeros((len(data_loader)*4,128,128))
                    
                    

                    outputs = self.model(inputs_left,inputs_right,feat_sam,flow_l2r)  
                    
                    
                else:
                    inputs= data.to(device=self.device, non_blocking=True)
                    gt_mask = target.to(device=self.device, non_blocking=True)
                    feat_sam = feat_sam.to(device=self.device, non_blocking=True)                                              
                    outputs = self.model(inputs,feat_sam)  


                losses = self.criterion(outputs, gt_mask)
                
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
                # loss = 2*loss_ce + 0.75*loss_dice + 1*loss_mask
                
                
                # reduced_loss = reduce_value(loss, average=True)
                # reduced_loss_ce = reduce_value(loss_ce, average=True)
                # reduced_loss_dice = reduce_value(loss_dice, average=True)
                # reduced_loss_mask = reduce_value(loss_mask, average=True)
                
                losses_list_val.append(loss.item())
                loss_ce_list_val.append(loss_ce.item())
                loss_dice_list_val.append(loss_dice.item())
                loss_mask_list_val.append(loss_mask.item())
                index=eval_loader.batch_size
                # 生成预测
                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                
                # output=instance_inference_pure(mask_cls_results, mask_pred_results)
                # for i in range(len(output)):
                #     pred_masks_val = output[i]
                
                    # pred_masks_val = self.semantic_inference_topk(mask_cls_results, mask_pred_results,2)
                    # pred_masks_val = self.semantic_inference(mask_cls_results, mask_pred_results)   
                    # pred_masks_val=get_segimg(pred_masks_val)

                # pred_masks_val = semantic_inference_with_bg(mask_cls_results, mask_pred_results)
                # gt_mask =target[0]
                # gt_binary_mask = self._get_binary_mask_one(gt_mask)
                # # dice=dice_coeff(pred_masks, gt_binary_mask.to(self.device),num_cls=7,epsilon=1e-6) 
                # dice = self._get_dice(pred_masks_val[0], gt_binary_mask.to(self.device))
                
                # dice_score.append(reduce_value(dice).item())
                # print('dice:',np.mean(dice_score))
                # if self.cfg.inferonly:         
                #     pred_masks_val = self.semantic_inference_topk(mask_cls_results, mask_pred_results,2)
                # else:
                #     pred_masks_val = self.semantic_inference(mask_cls_results, mask_pred_results)   
                      
                # pred_masks_val=get_segimg(pred_masks_val)
                # val_pred[index*i:index*i+index,:,:] = pred_masks_val.argmax(dim=1)
               
                    
 
                    
                instance=True
                if instance:
                    for idx in range(len(mask_cls_results)):
                        pic_name = name[idx]
                        if pic_name != "seq_2_frame102":
                            mask_cls_instance= mask_cls_results[idx]
                            mask_pred_instance= mask_pred_results[idx]
                            pred_outputs = instance_inference_pure(
                                mask_cls_instance, mask_pred_instance,pic_name, 
                                test_topk_per_image=64, panoptic_on=False,
                                thing_class_ids=None, box_from_mask=False, mask_bin_thresh=0.0
                                )
                    continue
            
                            # sem_maps = batch_topk_instances_to_semantic_overwrite(
                            #         pred_outputs, num_classes=num_classes, topk=4, mask_thresh=0.5
                            #         )
                
                    # pred_outputs = instance_inference_pure(
                    #     mask_cls_results, mask_pred_results, 
                    #     test_topk_per_image=64, panoptic_on=False,
                    #     thing_class_ids=None, box_from_mask=False, mask_bin_thresh=0.0
                    # )
                    # # pred_outputs: 单图时为 dict；批量时为 list[dict]

                    # num_classes = 7  # 你的类别数（不含背景）

                    # # 得到语义分割图（单图 or 批量）
                    # sem_maps = batch_topk_instances_to_semantic_overwrite(
                    #     pred_outputs, num_classes=num_classes, topk=4, mask_thresh=0.5
                    # )
                    # val_pred[index*i:index*i+index,:,:] = torch.stack(sem_maps, dim=0) 
                else:
                    if self.cfg.inferonly:  
                        # pred_masks_val=semantic_inference_with_bg(mask_cls_results, mask_pred_results)
                        pred_masks_val = self.semantic_inference_topk(mask_cls_results, mask_pred_results,2)
                    else:
                        pred_masks_val = self.semantic_inference(mask_cls_results, mask_pred_results)   
                        
                    pred_masks_val=get_segimg(pred_masks_val)

                    val_pred[index*i:index*i+index,:,:] = pred_masks_val.argmax(dim=1)

                # for i in range(len(sem_maps)):
                #     vis(sem_maps[i],i)
                #     vis(gt_mask[i],'gt')
                #     print('haha')
                
                # for i in range(len(pred_masks_val)):
                #     scores_per_image, topk_idx= pred_masks_val[i]['scores'].topk(2)
                #     (scores_per_image, pred_masks_val[i]['pred_masks'], scores_per_image)
                #     for k in topk_idx:
                #         pred_masks= pred_masks_val[i]['pred_masks'][k].detach().cpu()*(pred_masks_val[i]['pred_classes'][k].detach().cpu()+1)
                #         vis(pred_masks,k)
                #     vis(gt_mask[i],'gt')
                #     print('haha')

                # pred_masks_val = self.semantic_inference(mask_cls_results, mask_pred_results) 

                # val_pred[index*i:index*i+index,:,:] = pred_masks_val
                 #list变tensor
                val_target[index*i:index*i+index,:,:] = gt_mask 
                names+=name 


            preds_all=val_pred
            targets_all=val_target
            dice_all=[]
            val_iou = eval_endovis(preds_all, targets_all, num_classes=7, ignore_background=True, device='cpu')
            if self.cfg.inferonly:
                for i in range(len(preds_all)):
                    dice_mean_new=dice_coeff(preds_all[i,:,:].unsqueeze(0), targets_all[i,:,:].unsqueeze(0).to(self.device))
                    dice_all.append(dice_mean_new.detach().cpu().numpy())
                    # 保存结果图
                    # visanddraw(preds_all[i,:,:], targets_all[i,:,:].to(self.device),names[i],dice_mean_new,'test4')
                
                dice=np.mean(dice_all)
                dice_mean = dice.mean().item()
            # dice_all = dice_coeff(preds_all, targets_all)
            
            score=val_iou['IoU'] #就是ISINET IOU，对mask求平均
            # print('IOU:',val_iou['IoU'])
            # 记录日志
            writer.add_scalar('val/val_loss', np.mean(losses_list_val), epoch)
            writer.add_scalar('val/val_dice_loss', np.mean(loss_dice_list_val), epoch)
            writer.add_scalar('val/val_bce_loss', np.mean(loss_ce_list_val), epoch)
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
            with open('/home/yjh/code_yjh_bishe/Mask2Former-Simplify-master/log_new/log_'+self.cfg.task+'.txt', 'a') as f:
                # f.write('val evaluate dice: {0}, IoU: {1},mIoU: {2}/n'.format(score))
            # with open('/home/gjy/code_yjh/Mask2Former-Simplify-master/log/log_endovis_2017_resnet50_1.txt', 'a') as f:
                
                # f.write('evaluate dice: {0},losses:{1}/n'.format(score,np.mean(losses_list_val)))  
                f.write('evaluate IIoU: {0},losses:{1}/n'.format(score,np.mean(losses_list_val)))
                progress = f'loss:{(np.mean(losses_list_val)):.6f} loss_ce:{(np.mean(loss_ce_list_val)):.6f} loss_dice:{(np.mean(loss_dice_list_val)):.6f} loss_mask:{(np.mean(loss_mask_list_val)):.6f}\n '
                f.write(progress) 
            del val_pred,val_target
        # else:
        #     score = torch.tensor(0.0) 
        # dist.barrier()     # 确保所有进程同步

        # # 广播主进程的指标到所有进程
        # score_tensor = torch.tensor(score if rank == 0 else 0.0, device=self.device)
        # dist.broadcast(score_tensor, src=0)
        # score = score_tensor.item()
        # del gathered_preds ,gathered_targets
        
        return score

    # @torch.no_grad() 
    # def evaluate_sample(self):        
    #     nuim = NuImages(dataroot=self.cfg.DATASETS.ROOT_DIR, version='v1.0-test') # v1.0-test or v1.0-mini
    #     sample_idx_list = np.random.choice(len(nuim.sample), 10, replace=False)
    #     seg_handler = Segmentation(self.cfg, self.model)
    #     input_imgs = []
    #     render_imgs = []
    #     for idx in sample_idx_list:
    #         sample = nuim.sample[idx]
    #         sd_token = sample['key_camera_token']
    #         sample_data = nuim.get('sample_data', sd_token)
            
    #         im_path = os.path.join(nuim.dataroot, sample_data['filename'])
    #         input_imgs.append(im_path)
    #     preds = seg_handler.forward(input_imgs)
    #     for i, img_path in enumerate(input_imgs):
    #         img = Image.open(img_path)
    #         render_img = nuim.render_predict(img, preds[i])
    #         render_imgs.append(render_img)
    #     return render_imgs

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
        return target_onehot[1:]


    def semantic_inference(self, mask_cls, mask_pred):       
        mask_cls = F.softmax(mask_cls, dim=-1)[...,1:] #(batchsize, num_queries, num_classes) ,值表示每个类别的概率
        # print( mask_cls[0])#此处对0-7类别（考虑背景）做softmax，之后才可以用argmax得到类别
        mask_pred = mask_pred.sigmoid()   #(batchsize, num_queries, h, w) ,值表示每个像素属于每个mask的概率(即不属于背景的概率)
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred) 
        return semseg
    

    

    
    def semantic_inference_topk(self, mask_cls, mask_pred,topk=2):  
      
          # 初始处理
        mask_cls = F.softmax(mask_cls, dim=-1)[..., 1:]# (B, Q, C)
        mask_pred = mask_pred.sigmoid() 
        mask_pred=nn.Threshold(0.1,0)(mask_pred)# (B, Q, H, W)
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
            print(valid_mask[b])
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


    # # 实例分割待调试
    # def instance_inference(self, mask_cls, mask_pred):
    #     # mask_pred is already processed to have the same shape as original input
    #     image_size = mask_pred.shape[-2:]

    #     # [Q, K]
    #     scores = F.softmax(mask_cls, dim=-1)[:, :-1]
    #     labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
    #     # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
    #     scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
    #     labels_per_image = labels[topk_indices]

    #     topk_indices = topk_indices // self.sem_seg_head.num_classes
    #     # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
    #     mask_pred = mask_pred[topk_indices]

    #     # if this is panoptic segmentation, we only keep the "thing" classes
    #     if self.panoptic_on:
    #         keep = torch.zeros_like(scores_per_image).bool()
    #         for i, lab in enumerate(labels_per_image):
    #             keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

    #         scores_per_image = scores_per_image[keep]
    #         labels_per_image = labels_per_image[keep]
    #         mask_pred = mask_pred[keep]

    #     result = Instances(image_size)
    #     # mask (before sigmoid)
    #     result.pred_masks = (mask_pred > 0).float()
    #     result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
    #     # Uncomment the following to get boxes from masks (this is slow)
    #     # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

    #     # calculate average mask prob
    #     mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
    #     result.scores = scores_per_image * mask_scores_per_image
    #     result.pred_classes = labels_per_image
    #     return result
    
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