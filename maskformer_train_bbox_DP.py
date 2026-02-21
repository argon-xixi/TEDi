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
from torchvision import transforms
import sys
sys.path.append('/home/gjy/code_yjh/pytorch-grad-cam-master')
import math
import itertools
from PIL import Image
# import wandb
from Data import dataloaders
from modeling.MaskFormerModel import MaskFormerModel
from modeling.MaskFormerModel_bina import MaskFormerModel_bina
from modeling.MaskFormerModel_bbox import MaskFormerModel_bbox

from utils.criterion_bbox import SetCriterion_bbox
from utils.matcher import HungarianMatcher
from utils.matcher_bbox import HungarianMatcher_bbox
from utils.summary import create_summary
from utils.solver import maybe_add_gradient_clipping
from utils.misc import load_parallal_model
# from dataset.NuImages import NuImages
from Segmentation import Segmentation
from yjh.mIOU_new import eval_endovis, eval_endovis_bina
import cv2

import psutil
# from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
# from models.model_stages_double import BiSeNet
# from pytorch_grad_cam.grad_cam import GradCAM

def vis(pred_mask,gt,name):
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
        
        # 使用NumPy数组索引获取颜色
        
        # zn3=torch.mul(pred_mask,target).argmax(dim=0)
        # zn3=zn3.detach().cpu().numpy().astype(np.uint8)
        # zn3 = [x + 1 if x != 0 else x for x in zn3]
        # color_mask1 = palette[zn3]
        color_mask1 = palette[z_n1]  # 自动广播到 (H, W, 3)
        
        # # 转换颜色空间并保存
        # cv2.imwrite(
        #     f'/home/gjy/code_yjh/Mask2Former-Simplify-master/test1/pred/{i}_{dice}.png',
        #     cv2.cvtColor(color_mask1, cv2.COLOR_RGB2BGR)
        # )
        # print('保存成功')

        # ================ 可视化真实标签 ================
        
        # resize=transforms.Resize((128,128))
        # gt=resize(gt)
        # gt= gt.permute(1,2,0)*255
        gt = gt.detach().cpu().numpy().astype(np.uint8)
        # gt= gt
        # 检查数值范围
        gt = np.clip(gt, 0, len(palette)-1)
        
        # 获取颜色映射
        color_mask2 = palette[gt]
        img_flo = np.concatenate([color_mask1, color_mask2], axis=0)
        cv2.imwrite('/home/gjy/code_yjh/Mask2Former-Simplify-master/yjh/pred_img/'+name+'.png',img_flo )

def print_memory_usage():
    process = psutil.Process()
    print(f"Memory used: {process.memory_info().rss / 1024 ** 2:.2f} MB")
    
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

class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
 
    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()

class MaskFormer_bbox():
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.num_queries = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        self.size_divisibility = cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY
        self.num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        if cfg.DDP==True:
            self.device = torch.device("cuda", cfg.local_rank)
        else:
            self.device = torch.device('cuda:0')
        self.is_training = cfg.MODEL.IS_TRAINING
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.last_lr = cfg.SOLVER.LR
        self.start_epoch = 0
        if cfg.bbox:
            if cfg.bina:
                self.model = MaskFormerModel_bina(cfg)
            else:
                
                self.model = MaskFormerModel_bbox(cfg)
            
        else:
                
            if cfg.bina:
                self.model = MaskFormerModel_bina(cfg)
            else:
                self.model = MaskFormerModel(cfg)
        
        if cfg.MODEL.PRETRAINED_WEIGHTS is not None and os.path.exists(cfg.MODEL.PRETRAINED_WEIGHTS):
            self.load_model(cfg.MODEL.PRETRAINED_WEIGHTS)
            print("loaded pretrain mode:{}".format(cfg.MODEL.PRETRAINED_WEIGHTS))

       
        if cfg.ngpus > 1:
            if cfg.DDP==True:
                self.model = self.model.to(self.device)
                self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[cfg.local_rank], output_device=cfg.local_rank) 
            else: 
                
                self.model = self.model.to(self.device) 
                if not cfg.inferonly:
                    self.model=nn.DataParallel(self.model,device_ids=[0,1,2,3])    
                else:
                    self.model=nn.DataParallel(self.model,device_ids=[0,1])         

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

    def load_model(self, pretrain_weights):
        state_dict = torch.load(pretrain_weights)
        
        print('loaded pretrained weights form %s !' % pretrain_weights)
        dict_new={}
        print(self.model.state_dict().keys())
        ckpt_dict = state_dict['model']
        print(ckpt_dict.keys())
        if self.cfg.bina:
            
            for k, v in ckpt_dict.items():
                if k.startswith('module.'):
                    # if 'backbone' in k:
                    #     k1=k[7:].replace('backbone','backbone1')
                    #     k2=k[7:].replace('backbone','backbone2')
                    #     dict_new[k1] = v
                    #     dict_new[k2] = v
                    # else:
                    #     dict_new[k[7:]] = v
                    dict_new[k[7:]] = v
                else:
                    dict_new[k] = v
        else:
            
            for k, v in ckpt_dict.items():
                if k.startswith('module.'):
                    
                    
                        dict_new[k[7:]] = v
                else:
                    dict_new[k] = v
            
            
        print(dict_new.keys())
        self.last_lr = 6e-5
        # self.last_lr = 6e-5 # state_dict['lr']
        self.start_epoch = 70 # state_dict['epoch']
        self.model = load_parallal_model(self.model, dict_new)
        print('haha')

    def _training_init(self, cfg):
        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        bbox_weight = cfg.MODEL.MASK_FORMER.BBOX_WEIGHT
        giou_weight = cfg.MODEL.MASK_FORMER.GIOU_WEIGHT
        boundary_weight = cfg.MODEL.MASK_FORMER.BOUNDARY_WEIGHT

        # building criterion
       
        matcher = HungarianMatcher_bbox(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            cost_bbox=bbox_weight,
            cost_giou=giou_weight,
            
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )
            
        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight,"loss_bbox":bbox_weight,"loss_giou":giou_weight}
        
        
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        
        losses = ["labels", "masks","bbox"]
        self.criterion = SetCriterion_bbox(
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
        
        # TrainPath=[]
        # ValPath=[]
        # # for i in range(0,9):
        # #     index=0
        # #     for k in train_paths_temp:
                
                
        # #         if 'dataset_'+str(i) in k:
        # #             index+=1
        # #             if index< 169 :
        # #                 TrainPath.append(k)
        # #             else:
        # #                  ValPath.append(k)
                         
                
            
        # for i in train_paths_temp:
        #     if 'dataset_6' in i or 'dataset_7' in i:
        #         ValPath.append(i)
        #     else:
        #         TrainPath.append(i)
        
        # shuffled_list = copy.deepcopy(train_paths_temp)
        # random.seed(42)
        # random.shuffle(shuffled_list)

        # train = shuffled_list[:1350]

        # test = shuffled_list[1350:]
        max_score = 0.5
        for epoch in range(self.start_epoch + 1, n_epochs):
            # epoch+=15
            # epoch=82
            if epoch % 2 == 0  :
                version = 0 
            else:
                version = int((epoch % 48 + 1)/2)
                # version = 0 
           
            train_dataloader, val_dataloader = dataloaders.get_dataloaders(
            train_paths_temp,test_paths_temp, self.cfg,batch_size=self.cfg.batch_size,num_workers=self.cfg.workers,version=version,bina=self.cfg.bina)
            # train_paths_temp,test_paths_temp, batch_size=self.cfg.batch_size,num_workers=self.cfg.workers,version=version,bina=False)

            if not self.cfg.inferonly:
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
            del train_dataloader ,val_dataloader
            
            # torch.distributed.barrier() 
            self.summary_writer.close()




    def train_epoch(self,data_loader, epoch,writer):
        
        self.model.train()
        self.criterion.train()
        load_t0 = time.time()
        
      
        dice_score = []
        losses_list_train = []
        loss_ce_list_train = []
        loss_dice_list_train = []
        loss_mask_list_train = []

        loss_bbox_list_train = []
        loss_giou_list_train = []
        
        
        for i, (data, targets,feat_sam,name) in enumerate(data_loader):     
            if i % 10 == 0:
                print_memory_usage()                
            # inputs = batch['images'].to(device=self.device, non_blocking=True)
            # targets = batch['masks']
            # print(data.max())
            train_iou={}
            
           
            image_left= data[0].to(device=self.device, non_blocking=True)
            
            target = targets[0].to(device= self.device,non_blocking=True)
            bboxes = targets[1].to(device= self.device,non_blocking=True)
            cls = targets[2].to(device= self.device,non_blocking=True)
            feat_sam = feat_sam.to(device= self.device,non_blocking=True)
            if self.cfg.bina==True:
                image_right= data[1].to(device= self.device,non_blocking=True)
                flow_l2r= data[2].to(device= self.device,non_blocking=True)
                # print(f"Input device: {image_left.device}")
                # print(f"Model parameters device: {next(self.model.parameters()).device}")
                outputs = self.model(image_left,image_right,feat_sam,flow_l2r) 
            else:
                outputs = self.model(image_left,feat_sam)
            losses = self.criterion(outputs, target,bboxes,cls)
            

            weight_dict = self.criterion.weight_dict           
            loss_ce = 0.0
            loss_dice = 0.0
            loss_mask = 0.0
            loss_bbox = 0.0
            loss_giou = 0.0
                
            for k in list(losses.keys()):
                if k in weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                    if '_ce' in k:
                        loss_ce += losses[k]
                    elif '_dice' in k:
                        loss_dice += losses[k]
                    elif '_mask' in k:
                        loss_mask += losses[k]
                    elif '_bbox' in k:
                        loss_bbox += losses[k]
                    elif '_giou' in k:
                        loss_giou += losses[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            bbox_weight=0.1
            loss = loss_ce + loss_dice + loss_mask+ bbox_weight*(loss_bbox + loss_giou)
            
            # if reduce_value(loss_bbox).item()>1000:
            #     pass
            self.model.zero_grad()
            self.criterion.zero_grad()
            loss.backward()
            
            losses_list_train.append(loss.item())
            loss_ce_list_train.append(loss_ce.item())
            loss_dice_list_train.append(loss_dice.item())
            loss_mask_list_train.append(loss_mask.item())
            loss_bbox_list_train.append(loss_bbox.item())
            loss_giou_list_train.append(loss_giou.item())
                
            # with torch.no_grad():
            #     losses_list_train.append(reduce_value(loss).item())
            #     loss_ce_list_train.append(reduce_value(loss_ce).item())
            #     loss_dice_list_train.append(reduce_value(loss_dice).item())
            #     loss_mask_list_train.append(reduce_value(loss_mask).item())
            #     loss_bbox_list_train.append(reduce_value(loss_bbox).item())
            #     loss_giou_list_train.append(reduce_value(loss_giou).item())
            
            # loss = self.reduce_mean(loss, dist.get_world_size())

            
            # loss = self.reduce_mean(loss, dist.get_world_size())
            self.optim.step()
            # if dist.get_rank() == 0:
            elapsed = int(time.time() - load_t0)
            eta = int(elapsed / (i + 1) * (len(data_loader) - (i + 1)))
            curent_lr = self.optim.param_groups[0]['lr']
            # progress = f'\r[train] {i + 1}/{len(data_loader)} epoch:{epoch} {elapsed}(s) eta:{eta}(s) loss:{(np.mean(losses_list_train)):.6f} loss_ce:{(np.mean(loss_ce_list_train)):.6f} loss_dice:{(np.mean(loss_dice_list_train)):.6f} loss_mask:{(np.mean(loss_mask_list_train)):.6f}, lr:{curent_lr:.2e} '
            # # progress = f'\r[train] {i + 1}/{len(data_loader)} epoch:{epoch} {elapsed}(s) eta:{eta}(s) loss:{(np.mean(losses_list)):.6f} loss_ce:{(np.mean(loss_ce_list)):.6f} loss_dice:{(np.mean(loss_dice_list)):.6f}, lr:{curent_lr:.2e}  '
            # print(progress, end=' ')
            sys.stdout.flush()   
            
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]            
            pred_masks = self.semantic_inference(mask_cls_results, mask_pred_results)  
            batch_size = target.shape[0]
            gts=self.instance_mask2semantic_mask_batch(target,cls)
            dice_scores = []
            for j in range(batch_size):
                gt_binary_mask = self._get_binary_mask_one(gts[j])
                dice = self._get_dice(pred_masks[j], gt_binary_mask.to(self.device))
                dice_scores.append(dice)
            batch_dice = torch.mean(torch.stack(dice_scores))
            dice_score.append(batch_dice.detach().cpu().numpy())
                # dice_score.append(reduce_value(batch_dice, average=True).item())   
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

            # elapsed = int(time.time() - load_t0)
            # eta = int(elapsed / (i + 1) * (len(data_loader) - (i + 1)))
            # curent_lr = self.optim.param_groups[0]['lr']
           
            progress1 = f'\r[train] {i + 1}/{len(data_loader)} epoch:{epoch} {elapsed}(s) eta:{eta}(s) loss:{(np.mean(losses_list_train)):.6f} loss_ce:{(np.mean(loss_ce_list_train)):.6f} loss_dice:{(np.mean(loss_dice_list_train)):.6f} loss_mask:{(np.mean(loss_mask_list_train)):.6f}, lr:{curent_lr:.2e},dice:{(np.mean(dice_score)):.6f}\n '
            progress2 = f'\r[train] {i + 1}/{len(data_loader)} epoch:{epoch} {elapsed}(s) eta:{eta}(s)  loss_bbox:{(np.mean(loss_bbox_list_train)):.6f} loss_giou:{(np.mean(loss_giou_list_train)):.6f} \n '
        
            # # progress = f'\r[train] {i + 1}/{len(data_loader)} epoch:{epoch} {elapsed}(s) eta:{eta}(s) loss:{(np.mean(losses_list)):.6f} loss_ce:{(np.mean(loss_ce_list)):.6f} loss_dice:{(np.mean(loss_dice_list)):.6f}, lr:{curent_lr:.2e}  '
            print(progress1, end=' ')
            print(progress2, end=' ')
                
                
            with open('/home/gjy/code_yjh/Mask2Former-Simplify-master/log/log_'+self.cfg.task+'.txt', 'a') as f:
                f.write(progress1)
                f.write(progress2)
            sys.stdout.flush()
       
        # train_iou=eval_endovis(train_pred,train_target)
        writer.add_scalar('train/total_loss', np.mean(losses_list_train), epoch)
        writer.add_scalar('train/dice_loss', np.mean(loss_dice_list_train), epoch)
        writer.add_scalar('train/bce_loss', np.mean(loss_ce_list_train), epoch)
        writer.add_scalar('train/mask_loss', np.mean(loss_mask_list_train), epoch)
        writer.add_scalar('train/mask_loss', np.mean(loss_giou_list_train), epoch)
        writer.add_scalar('train/train_loss', np.mean( loss_bbox_list_train), epoch)
        writer.add_scalar('train/train_dice', np.mean(dice_score), epoch)
        # writer.add_scalar('train/train_challengIoU', train_iou['challengIoU'], epoch)
        # writer.add_scalar('train/train_IoU', train_iou['IoU'], epoch)
        # writer.add_scalar('train/train_mcIoU', train_iou['mcIoU'], epoch)
        # writer.add_scalar('train/train_mIoU', train_iou['mIoU'], epoch)
        # for i in range(len(train_iou['cIoU_per_class'])):
        #     per_class_train['cIoU_per_class_'+str(i+1)]=torch.tensor(train_iou['cIoU_per_class'][i])                
        
       
        
       
        return loss.item()

    @torch.no_grad()                   
    def evaluate(self, eval_loader,epoch,writer):
        
        self.model.eval()
        val_iou={}
        self.criterion.eval()
        dice_score = []
        losses_list_val = []
        loss_ce_list_val = []
        loss_dice_list_val = []
        loss_mask_list_val = []
        loss_bbox_list_val = []
        loss_giou_list_val = []
        per_class_val={}
         # 用于收集各进程的预测和目标
        preds_list = []
        targets_list = []
        per_class_val={}
       
        val_pred=torch.zeros((len(eval_loader.dataset),128,128)).to(device=self.device)
        val_target=torch.zeros((len(eval_loader.dataset),128,128)).to(device=self.device)
        
        with torch.no_grad():
            for i, (data, targets,feat_sam,name) in enumerate(eval_loader):
                
                
                
                target = targets[0].to(device=self.device, non_blocking=True)
                bboxes = targets[1].to(device=self.device, non_blocking=True)
                cls = targets[2].to(device=self.device, non_blocking=True)
                feat_sam = feat_sam.to(device=self.device, non_blocking=True)
                if self.cfg.bina==True:
                    image_left= data[0].to(device=self.device, non_blocking=True)
                    image_right= data[1].to(device=self.device, non_blocking=True)
                    flow_l2r= data[2].to(device=self.device, non_blocking=True)
                    outputs = self.model(image_left,image_right,feat_sam,flow_l2r) 
                else:
                    image_left= data[0].to(device=self.device, non_blocking=True)
                    outputs = self.model(image_left,feat_sam)
                losses = self.criterion(outputs, target,bboxes,cls)
                weight_dict = self.criterion.weight_dict           
                loss_ce = 0.0
                loss_dice = 0.0
                loss_mask = 0.0
                loss_bbox = 0.0
                loss_giou = 0.0
                for k in list(losses.keys()):
                    if k in weight_dict:
                        losses[k] *= self.criterion.weight_dict[k]
                        if '_ce' in k:
                            loss_ce += losses[k]
                        elif '_dice' in k:
                            loss_dice += losses[k]
                        elif '_mask' in k:
                            loss_mask += losses[k]
                        elif '_bbox' in k:
                            loss_bbox += losses[k]
                        elif '_giou' in k:
                            loss_giou += losses[k]
                    else:
                        # remove this loss if not specified in `weight_dict`
                        losses.pop(k)
                bbox_weight=0.5
                loss = loss_ce + loss_dice + loss_mask+ bbox_weight*(loss_bbox + loss_giou)
                
                losses_list_val.append(loss.item())
                loss_ce_list_val.append(loss_ce.item())
                loss_dice_list_val.append(loss_dice.item())
                loss_mask_list_val.append(loss_mask.item())
                loss_bbox_list_val.append(loss_bbox.item())
                loss_giou_list_val.append(loss_giou.item())
                
                
               
                
                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                if self.cfg.inferonly:
                    pred_masks_val = self.semantic_inference(mask_cls_results, mask_pred_results)
                    # pred_masks_val =F.softmax(pred_masks_val, dim=1)
                else:
                    pred_masks_val = self.semantic_inference(mask_cls_results, mask_pred_results)      
                dice_score1=[]
                index=len(name)     
                
                # pred_masks_val=get_segimg(pred_masks_val)
                # 收集当前批次的预测和目标
                # preds_list.append(pred_masks_val.argmax(dim=1)) 
                
                gts=  self.instance_mask2semantic_mask_batch(target,cls)  
                # print(gts.max())
            
                pred_masks_val=get_segimg(pred_masks_val)
                if self.cfg.inferonly:
                    for i in range(len(pred_masks_val)):
                        vis(pred_masks_val.argmax(dim=1)[i,:,:],gts [i,:,:],name[i])
                val_pred[index*i:index*i+index,:,:] = pred_masks_val.argmax(dim=1)
                # val_pred[index*i:index*i+index,:,:,:] = pred_masks_val
                val_target[index*i:index*i+index,:,:] = gts  
                
                    
           
 
            
            dice_all=[]
            val_iou = eval_endovis(val_pred, val_target, num_classes=7, ignore_background=True, device='cpu')
            if self.cfg.inferonly:
                for i in range(len(val_pred)):
                    dice_mean_new=dice_coeff(val_pred[i,:,:].unsqueeze(0), val_target[i,:,:].unsqueeze(0).to(self.device))
                    dice_all.append(dice_mean_new.detach().cpu().numpy())
                dice=np.mean(dice_all)
            
            # dice_all = dice_coeff(preds_all, targets_all)
            # score = dice_all.mean().item()
            score=val_iou['IoU'] #ISINET IOU 78.965
            # print('IOU:',val_iou['IoU'])
            # 记录日志
            writer.add_scalar('val/val_loss', np.mean(losses_list_val), epoch)
            writer.add_scalar('val/val_dice_loss', np.mean(loss_dice_list_val), epoch)
            writer.add_scalar('val/val_bce_loss', np.mean(loss_ce_list_val), epoch)
            writer.add_scalar('val/val_bbox_loss', np.mean(loss_bbox_list_val), epoch)
            writer.add_scalar('val/val_giou_loss', np.mean(loss_giou_list_val), epoch)
            writer.add_scalar('val/val_challengIoU', val_iou['challengIoU'], epoch)
            writer.add_scalar('val/val_IoU', val_iou['IoU'], epoch)  #对计算iou后每个mask平均
            writer.add_scalar('val/val_mcIoU', val_iou['mcIoU'], epoch)
            writer.add_scalar('val/val_mIoU', val_iou['mIoU'], epoch) #对每张图像平均（无视图像中mask数量的差异，每帧图像的贡献平等，无论其包含的像素数量或类别数量多少）
            for i in range(len(val_iou['cIoU_per_class'])):
                per_class_val['cIoU_per_class_'+str(i+1)]=torch.tensor(val_iou['cIoU_per_class'][i])
            writer.add_scalars('val/val_cIoU_per_class', per_class_val, epoch)
            
            
            # writer.add_scalar('val/val_dice', dice_all.mean().item(), epoch)
            
            with open('/home/gjy/code_yjh/Mask2Former-Simplify-master/log/log_'+self.cfg.task+'.txt', 'a') as f:
                # f.write('val evaluate dice: {0}, IoU: {1},mIoU: {2}/n'.format(score))
            # with open('/home/gjy/code_yjh/Mask2Former-Simplify-master/log/log_endovis_2017_resnet50_1.txt', 'a') as f:
                
                # f.write('evaluate dice: {0},losses:{1}/n'.format(score,np.mean(losses_list_val)))  
                f.write('evaluate IIoU: {0},losses:{1}/n'.format(score,np.mean(losses_list_val)))
                progress = f'loss:{(np.mean(losses_list_val)):.6f} loss_ce:{(np.mean(loss_ce_list_val)):.6f} loss_dice:{(np.mean(loss_dice_list_val)):.6f} loss_mask:{(np.mean(loss_mask_list_val)):.6f} loss_bbox:{(np.mean(loss_bbox_list_val)):.6f} loss_giou:{(np.mean(loss_giou_list_val)):.6f} \n '
                f.write(progress) 
                print('evaluate IIoU: {0},losses:{1}/n'.format(score,np.mean(losses_list_val)))
                print(progress)
            del val_pred,val_target
            return score
    
    
   
    def semantic_inference_topk(self, mask_cls, mask_pred,topk=2):  
      
          # 初始处理
        mask_cls = F.softmax(mask_cls, dim=-1)[..., 1:]# (B, Q, C)
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
        mask_pred = mask_pred.sigmoid()
        mask_pred=nn.Threshold(0.5,0)(mask_pred)#(batchsize, num_queries, h, w) ,值表示每个像素属于每个mask的概率(即不属于背景的概率)
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred) 
            
        return semseg
    
    def instance_mask2semantic_mask_batch(self, target, cls):
        """
        批量处理版本：将实例掩码转换为语义分割掩码
        Args:
            target: 实例掩码张量，形状为 (B, N, H, W)
            cls: 类别标签张量，形状为 (B, N) 1~7
        Returns:
            语义分割掩码，形状为 (B, H, W)
        """
       
        # overlap_mask=(target.sum(dim=1)>1)
        # a=target.sum(dim=1)
        # print(a.max())
        # 扩展维度用于广播计算
        cls_expanded = cls.unsqueeze(-1).unsqueeze(-1)  # 形状变为 (B, N, 1, 1)
        
        # 计算每个实例的贡献
        contributions = target * cls_expanded  # 广播乘法得到 (B, N, H, W)
        
        # 沿着实例维度求和
        semantic_mask,max_indice = torch.max(contributions,dim=1)  # 输出形状 (B, H, W) 由于不会出现不同类别的重复。故只需取最大就可以避免重复
        
        return semantic_mask
            

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