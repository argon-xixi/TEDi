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
from yjh.utils_yjh import instance_inference_pure,batch_topk_instances_to_semantic_overwrite,visanddraw,semantic_inference_with_bg,compute_iou_and_dice,plot_query_semseg_distribution
import einops
import json
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

class MaskFormer_baseline_infer():
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
                self.model=nn.DataParallel(self.model,device_ids=[0,1])  
        self._training_init(cfg) 
            # self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[cfg.local_rank], output_device=cfg.local_rank, find_unused_parameters=True)             

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

        self.last_lr = 4e-5
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
        self.save_folder = os.path.join(cfg.TRAIN.CKPT_DIR,cfg.task)
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
  
        if self.cfg.dataset == 'EndoVis2017':
            for i in train_paths_temp:
                if 'dataset_6' in i or 'dataset_7' in i:
                    ValPath.append(i)
                else:
                    TrainPath.append(i)
        max_score = 0.5
        for epoch in range(self.start_epoch + 1, n_epochs):
            if epoch % 2 == 0  :
                version = 0 
            else:
                version = 0
                # version = int((epoch % 64 + 1)/2)
                
            if self.cfg.dataset == 'EndoVis2017':
                train_dataloader, val_dataloader = dataloaders.get_dataloaders(
                    TrainPath,ValPath, self.cfg,batch_size=self.cfg.batch_size,num_workers=self.cfg.workers,version=version,bina=self.cfg.bina,)
            elif self.cfg.dataset == 'EndoVis2018':
                train_dataloader, val_dataloader = dataloaders.get_dataloaders(
                    train_paths_temp, test_paths_temp, self.cfg,batch_size=self.cfg.batch_size,num_workers=self.cfg.workers,version=version,bina=self.cfg.bina,)
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

            self.summary_writer.close()




   
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
        val_image=torch.zeros((len(eval_loader.dataset),128,128,3)).to(device=self.device)
        
        # val_mask_cls_results=torch.zeros((len(eval_loader.dataset),100,9)).to(device=self.device)
        # val_mask_pred_results=torch.zeros((len(eval_loader.dataset),100,128,128)).to(device=self.device)
        names=[]
 
        with torch.no_grad():
            for i, (data, target,feat_sam,name) in enumerate(eval_loader):
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

                losses,indice_dict = self.criterion(outputs, gt_mask)
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

                        losses.pop(k)
                loss = loss_ce + loss_dice + loss_mask
                
                losses_list_val.append(loss.item())
                loss_ce_list_val.append(loss_ce.item())
                loss_dice_list_val.append(loss_dice.item())
                loss_mask_list_val.append(loss_mask.item())
                
                # 生成预测
                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                gt_mask[gt_mask>7] -= 8
                instance=True   
                if instance:
                    pred_masks_val_list = []
                    
                    for idx in range(len(mask_cls_results)):
                        pic_name = name[idx]
                        if not pic_name == "seq_2_frame000":
                            continue
                        mask_cls_instance= mask_cls_results[idx]
                        mask_pred_instance= mask_pred_results[idx]
                        label = gt_mask[idx].max()
                        pred_outputs = instance_inference_pure(
                            "all",
                            mask_cls_instance, mask_pred_instance,pic_name, 
                            test_topk_per_image=10, panoptic_on=False,
                            thing_class_ids=None, box_from_mask=False, mask_bin_thresh=0.0,
                            draw=True
                            )
                        pred_masks_val_list.append(batch_topk_instances_to_semantic_overwrite(pic_name,pred_outputs,7,5))
                        exit()
                    continue
                    pred_masks_val=torch.stack(pred_masks_val_list,dim=0)
                    # print(batch_topk_instances_to_semantic_overwrite(pred_outputs,7,7).max())
                    index=eval_loader.batch_size
                    val_pred[index*i:index*i+index,:,:] = pred_masks_val
                    val_target[index*i:index*i+index,:,:] = gt_mask  #取后一帧

                    names+=name 
                else:
                # # # #         continue
                    if self.cfg.inferonly:
                        pred_masks_val = self.semantic_inference(mask_cls_results, mask_pred_results)
                        # pred_masks_val = self.semantic_inference_topk(mask_cls_results, mask_pred_results,2)
                    else:
                        pred_masks_val = self.semantic_inference(mask_cls_results, mask_pred_results)
                    index=eval_loader.batch_size      
                  

                    # val_pred[index*i:index*i+index,:,:] = pred_masks_val.argmax(dim=1)
                    if self.cfg.inferonly:
                        inputs = F.interpolate(inputs, size=(128, 128), mode="bilinear", align_corners=False)
                        x_denorm = denorm_bchw(inputs,  [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]).clamp(0, 1)*255
                        val_image[index*i:index*i+index,:,:,:] = x_denorm.permute(0,2,3,1)
                        class_ids = [0,1,2,3,4,5,6,7]
                        class_names = ["0","1","2","3","4","5","6","7"]
                        # for i in range(len(mask_cls_results)):
                        #     if name[i] == "seq_15_frame129":
                        #         index=eval_loader.batch_size      
                        #         # pred_masks_val=get_segimg(pred_masks_val)
                        #         pred_masks_val=pred_masks_val
                        #         # val_mask_cls_results[index*i:index*i+index,:,:] = mask_cls_results
                        #         # val_mask_pred_results[index*i:index*i+index,:,:,:] = mask_pred_results 
                        #         fig = plot_query_semseg_distribution(
                        #         gt=gt_mask[i].unsqueeze(0),
                        #         class_ids=class_ids,
                        #         class_names=class_names,
                        #         base=None,   # 没有就传 None
                        #         ours=(mask_cls_results[i].unsqueeze(0), mask_pred_results[0].unsqueeze(0)),   # 没有就传 None
                        #         ignore_index=255,
                        #         mode="log_prob",        # 推荐先用 "prob"（贴近你现在的 semseg），再试 "raw"
                        #         bins=220,
                        #         smooth_sigma=2.0,
                        #         max_pos=60000,
                        #         max_neg=60000,
                        #         seed=0,
                        #         share_x=True,
                        #         title="Comparison of Query-based SemSeg Score Distribution",
                        #         save_path="/home/yjh/code_yjh_bishe/Mask2Former-Simplify-master/yjh/query_semseg_dist_good.png"
                        #     )
                        #         exit()

                    
                    
                val_pred[index*i:index*i+index,:,:] = pred_masks_val.argmax(dim=1)
                val_target[index*i:index*i+index,:,:] = gt_mask 
                    
                names+=name 
    
            # # 计算二分类mask的dice和iou
            iou_mean, dice_mean = compute_iou_and_dice(val_pred, val_target)
            print("IoU mean:", iou_mean.item())
            print("Dice mean:", dice_mean.item())

            preds_all=val_pred
            targets_all=val_target
            
           
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
            
            
            dice_all=[]
            val_iou = eval_endovis(preds_all, targets_all, num_classes=7, ignore_background=True, device='cpu')
            if self.cfg.inferonly:
                for i in range(len(names)):
                    dice_mean_new=dice_coeff(preds_all[i,:,:].unsqueeze(0), targets_all[i,:,:].unsqueeze(0).to(self.device))
                    dice_all.append(dice_mean_new.detach().cpu().numpy())
                    visanddraw(preds_all[i,:,:], targets_all[i,:,:].to(self.device),names[i],dice_mean_new,'baseline_test1',val_image[i])
                dice=np.mean(dice_all)
            # dice_all = dice_coeff(preds_all, targets_all)
            # score = dice_all.mean().item()
            score=val_iou['IoU'] #就是ISINET IOU，对mask求平均
            print(np.mean(losses_list_val))
            # print('IOU:',val_iou['IoU'])
            # 记录日志
            # exit()
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
        return target_onehot[1:-1]


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
        mask_cls = F.softmax(mask_cls, dim=-1)[..., -1:]# (B, Q, C)
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