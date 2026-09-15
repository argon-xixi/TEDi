#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   maskformer3D.py
@Time    :   2022/09/30 20:50:53
@Author  :   BQH 
@Version :   1.0
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   DeformTransAtten分割网络训练代码
'''

import datetime
import itertools
import os
import pickle
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import distributed as dist
from torch import nn

from Data import dataloaders
from modeling.MaskFormerModel_tcd_mse import (
    MaskFormerModel_denoising_memorybank as MaskFormerModel_stage2,
)
from utils.misc import load_parallal_model
from utils.solver import maybe_add_gradient_clipping
from utils.summary import create_summary
from utils.evaluation import EvaluationMixin, dice_coeff


class tedi(EvaluationMixin):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_queries = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        self.size_divisibility = cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY
        self.num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        self.istrain =True
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.last_lr = cfg.SOLVER.LR
        self.start_epoch = 0
        if cfg.local_rank != -1:
            torch.cuda.set_device(cfg.local_rank)
            self.device=torch.device("cuda", cfg.local_rank)
        self.model = MaskFormerModel_stage2(cfg)
        self.model = self.model.to(self.device)
        if not cfg.inferonly:
            if cfg.MODEL.PRETRAINED_WEIGHTS is not None and os.path.exists(cfg.MODEL.PRETRAINED_WEIGHTS):
                self.load_state_dict_fix_numpy(cfg.MODEL.PRETRAINED_WEIGHTS)
                print("loaded pretrain mode:{}".format(cfg.MODEL.PRETRAINED_WEIGHTS))
        else:
            if cfg.MODEL.INFER_PRETRAINED_WEIGHTS is not None and os.path.exists(cfg.MODEL.INFER_PRETRAINED_WEIGHTS):
                self.load_state_dict_fix_numpy(cfg.MODEL.INFER_PRETRAINED_WEIGHTS)
                print("loaded pretrain mode:{}".format(cfg.MODEL.INFER_PRETRAINED_WEIGHTS))
        if cfg.ngpus > 1:
            if not cfg.inferonly:
                self.model=nn.DataParallel(self.model)    
            else:
                self.model=nn.DataParallel(self.model)   
          
        self._training_init(cfg)

    def _reset_memory(self):

        model = self.model
        if hasattr(model, "module"):
            model = model.module
        if hasattr(model, "memory"):
            model.memory.reset_memory()

    def build_optimizer(self):
        def maybe_add_full_model_gradient_clipping(optim):
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
        
        if isinstance(state_dict_temp, dict) and 'model' in state_dict_temp:
            state_dict_temp = state_dict_temp['model']
        
        new_state_dict = OrderedDict()
        for k, v in state_dict_temp.items():
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            new_state_dict[k] = v
        model_dict = self.model.state_dict()
        loaded_keys = []
        skipped_keys = []

        for k, v in new_state_dict.items():
            if "criterion" in k:
                print(k)
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    model_dict[k] = v
                    loaded_keys.append(k)
                else:
                    skipped_keys.append((k, v.shape, model_dict[k].shape))
            else:
                skipped_keys.append((k, v.shape, None))

        print(f"\n✅ Load Summary:")
        print(f"- Total model parameters       : {len(self.model.state_dict())}")
        print(f"- Successfully loaded          : {len(loaded_keys)}")
        print(f"- Skipped due to shape mismatch: {len(skipped_keys)}")
        
        if skipped_keys:
            print("\n❗Skipped keys due to mismatch or missing:")
            for k, shape_loaded, shape_model in skipped_keys:
                print(f"  - {k}: loaded {shape_loaded}, model expects {shape_model}")
                
        self.model.load_state_dict(model_dict, strict=False)
        
    def _training_init(self, cfg):
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        self.weight_dict = weight_dict
        self.summary_writer = create_summary(0, log_dir=cfg.TRAIN.LOG_DIR)
        self.save_folder = os.path.join(cfg.TRAIN.CKPT_DIR,cfg.task,str(cfg.fold))
        self.optim = self.build_optimizer()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode='max', factor=0.9, patience=10)

    def reduce_mean(self, tensor, nprocs): 
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= nprocs
        return rt

    def train(self, train_paths_temp, test_paths_temp, n_epochs,writer,fold):
        max_score = 0.5
        version = 0
        if self.cfg.dataset == 'EndoVis2017':
            train_dataloader, val_dataloader = dataloaders.get_dataloaders(
                train_paths_temp,test_paths_temp, self.cfg,batch_size=self.cfg.batch_size,num_workers=self.cfg.workers,version=version,bina=self.cfg.bina,)
        elif self.cfg.dataset == 'EndoVis2018':
            train_dataloader, val_dataloader = dataloaders.get_dataloaders(
                train_paths_temp, test_paths_temp, self.cfg,batch_size=self.cfg.batch_size,num_workers=self.cfg.workers,version=version,bina=self.cfg.bina,)
        for epoch in range(self.start_epoch + 1, n_epochs):
            if not self.cfg.inferonly:
                train_loss = self.train_epoch(train_dataloader, epoch,writer)
            evaluator_score = self.evaluate(val_dataloader,epoch,writer)
            self.scheduler.step(evaluator_score)
            os.makedirs(self.save_folder, exist_ok=True)
            max_score = evaluator_score
            ckpt_path = os.path.join(self.save_folder, 'mask2former_Epoch{0}_dice{1:.4f}.pth'.format(epoch, max_score))
            save_state = {'model': self.model.state_dict(),
                        'lr': self.optim.param_groups[0]['lr'],
                        'epoch': epoch}
            torch.save(save_state, ckpt_path)
            print('weights {0} saved success!'.format(ckpt_path))
            self.summary_writer.close()

    def train_epoch(self,data_loader,epoch,writer,warmup_only=False):
        sampler = getattr(data_loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        self.model.train()
        load_t0 = time.time()
        dice_score = []
        losses_list_train = []
        loss_ce_list_train = []
        loss_dice_list_train = []
        loss_mask_list_train = []
        self.istrain = True
        for i, (data, target,name) in enumerate(data_loader):  
            inputs= data.to(device=self.device, non_blocking=True)
            target = target.to(device=self.device, non_blocking=True)
            name_idx = torch.arange(len(name), device=self.device)
            outputs,losses= self.model(inputs,target,self.istrain,name_idx, epoch, i, name)  
            if hasattr(self.model, "module"):  
                self.model.module.iter += data_loader.batch_size
            else:
                self.model.iter += data_loader.batch_size
            loss_ce = 0.0
            loss_dice = 0.0
            loss_mask = 0.0
            for k in list(losses.keys()):
                if k in self.weight_dict:
                    losses[k] *= self.weight_dict[k]
                    if '_ce' in k:
                        loss_ce += losses[k]
                    elif '_dice' in k:
                        loss_dice += losses[k]
                    elif '_mask' in k:
                        loss_mask += losses[k]
                else:
                    losses.pop(k)

            loss = loss_ce + loss_dice + loss_mask
            loss=loss.mean()
            self.model.zero_grad()
            loss.backward()
            with torch.no_grad():
                losses_list_train.append(loss.mean().item())
                loss_ce_list_train.append(loss_ce.mean().item())
                loss_dice_list_train.append(loss_dice.mean().item())
                loss_mask_list_train.append(loss_mask.mean().item())
            self.optim.step()

            elapsed = int(time.time() - load_t0)
            eta = int(elapsed / (i + 1) * (len(data_loader) - (i + 1)))
            curent_lr = self.optim.param_groups[0]['lr']
            with torch.no_grad():
                outputs = self.post_processing(outputs)
                mask_cls_results = outputs["pred_logits"] 
                mask_pred_results = outputs["pred_masks"] 
                pred_masks = self.semantic_inference(mask_cls_results, mask_pred_results)
                target[target >7] -= 8
                dice_batch_mean =dice_coeff(pred_masks.argmax(1),target[:,-1,:,:])
                dice_score.append(dice_batch_mean.item())  
            progress = f'\r[train] {i + 1}/{len(data_loader)} epoch:{epoch} {elapsed}(s) eta:{eta}(s) loss:{(np.mean(losses_list_train)):.6f} loss_ce:{(np.mean(loss_ce_list_train)):.6f} loss_dice:{(np.mean(loss_dice_list_train)):.6f} loss_mask:{(np.mean(loss_mask_list_train)):.6f}, lr:{curent_lr:.2e},dice:{(np.mean(dice_score)):.6f}\n '
            print(progress, end=' ')
            
        writer.add_scalar('train/total_loss', np.mean(losses_list_train), epoch)
        writer.add_scalar('train/dice_loss', np.mean(loss_dice_list_train), epoch)
        writer.add_scalar('train/bce_loss', np.mean(loss_ce_list_train), epoch)
        writer.add_scalar('train/train_dice', np.mean(dice_score), epoch)
        
        return loss.item()
