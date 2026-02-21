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
from modeling.MaskFormerModel_feature_output import MaskFormerModel_baseline
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
import einops
from tqdm.auto import tqdm
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

class MaskFormer_feature_output():
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
                self.model=nn.DataParallel(self.model,device_ids=[0])    
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
        self.last_lr = 6e-5
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
            if 'criterion.' in k :
                k= k.replace('criterion.','')
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

        # test = shuffled_list[1350:]
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
            
            self.train_epoch(train_dataloader, epoch,writer)
            self.evaluate(val_dataloader,epoch,writer)
           

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
        for i, (data, target, feat_sam, name) in enumerate(
        tqdm(data_loader, desc="Train", unit="batch")):
                               
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
                inputs= data.to(device=self.device, non_blocking=True).squeeze(0)
                target = target.to(device=self.device, non_blocking=True).squeeze(0)
                torch.cuda.synchronize()
                t0 = time.time()
                # outputs = self.model(inputs)  
                # print(target.max())
                feature_name='train/'+name[0]+'.h5'
                outputs = self.model(inputs,target,feature_name) 
           
    
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
            for i, (data, target, feat_sam, name) in enumerate(
            tqdm(eval_loader, desc="Eval", unit="batch")):
                feature_name_list=[]
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
                    inputs= data.to(device=self.device, non_blocking=True).squeeze(0)
                    gt_mask = target.to(device=self.device, non_blocking=True).squeeze(0)
                    feature_name='test/'+name[0]+'.h5'
                    outputs = self.model(inputs,gt_mask,feature_name) 

               
    
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