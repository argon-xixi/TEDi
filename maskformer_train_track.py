

import os
import logging
import einops
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

from Data import dataloaders

from modeling.MaskFormerModel_track_memorybank import MaskFormerModel_track_memorybank as MaskFormerModel_track

from utils.criterion import SetCriterion, Criterion
from utils.matcher import HungarianMatcher
from utils.summary import create_summary
from utils.solver import maybe_add_gradient_clipping
from utils.misc import load_parallal_model

from utils.mIOU_new import eval_endovis
import cv2
import pickle
from collections import OrderedDict
from utils.utils import compute_iou_and_dice, overlay
import einops
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def denorm_bchw(x, mean, std):

    mean = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    std  = torch.tensor(std,  device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return x * std + mean

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
        "names": list(name_list),
        "indice_dict": indice_dict_to_serializable(indice_dict),
    }
    if extra is not None:
        record["extra"] = extra

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()

def reduce_value(value, average=True):
    world_size = dist.get_world_size()
    if world_size < 2:
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

    return value
def get_segimg(semseg):

    background_prob = 1 - semseg.sum(dim=1, keepdim=True)
    semseg_with_bg = torch.cat([background_prob, semseg], dim=1)
    pred_mask=semseg_with_bg

    return pred_mask
import torch
import torch.nn.functional as F

def _connected_components_instances(mask_hw: torch.Tensor, class_id: int, min_area: int = 1):

    m = (mask_hw == int(class_id)).detach().to("cpu").numpy().astype(np.uint8)
    if m.sum() == 0:
        return []
    num_labels, labels = cv2.connectedComponents(m, connectivity=8)
    out = []
    for k in range(1, num_labels):
        comp = (labels == k)
        area = int(comp.sum())
        if area >= int(min_area):
            out.append(torch.from_numpy(comp))
    return out

def build_gt_instances_from_semantic(gt_sem_hw: torch.Tensor, num_classes: int, min_area: int = 1):
    H, W = gt_sem_hw.shape
    gt_masks = []
    gt_labels = []
    for cid in range(1, num_classes + 1):
        comps = _connected_components_instances(gt_sem_hw, cid, min_area=min_area)
        for comp in comps:
            gt_masks.append(comp)
            gt_labels.append(cid)

    device = gt_sem_hw.device
    if len(gt_masks) == 0:
        masks_t = torch.zeros((0, H, W), dtype=torch.float32, device=device)
        labels_t = torch.zeros((0,), dtype=torch.int64, device=device)
    else:
        masks_t = torch.stack([m.to(device=device) for m in gt_masks], dim=0).float()
        labels_t = torch.as_tensor(gt_labels, dtype=torch.int64, device=device)
    return {"masks": masks_t, "labels": labels_t}

@torch.no_grad()
def eval_instance_class_and_mask(
    pred_logits_bqc: torch.Tensor,
    pred_masks_bqhw: torch.Tensor,
    gt_sem_bhw: torch.Tensor,
    num_classes: int = 8,
    score_thresh: float = 0.3,
    mask_thresh: float = 0.5,
    min_gt_area: int = 1,
    topk_preds: int = 30,
):

    from utils.matcher import HungarianMatcher

    B, Q, C1 = pred_logits_bqc.shape
    assert C1 == num_classes + 1, f"pred_logits last dim should be num_classes+1={num_classes+1}, got {C1}"

    prob = pred_logits_bqc.softmax(-1)
    prob_fg = prob[..., :num_classes]
    scores, pred_labels0 = prob_fg.max(-1)
    pred_labels = pred_labels0 + 1

    totals = {
        "n_gt": 0,
        "n_pred": 0,
        "n_match": 0,
        "class_correct": 0,
        "iou_sum": 0.0,
        "dice_sum": 0.0,
    }

    matcher = HungarianMatcher(cost_class=1.0, cost_mask=1.0, cost_dice=1.0, num_points=0)

    for b in range(B):
        gt_sem_hw = gt_sem_bhw[b]

        if gt_sem_hw.max() > num_classes:
            gt_sem_hw = gt_sem_hw.clone()
            gt_sem_hw[gt_sem_hw > num_classes] -= (num_classes + 1)

        tgt = build_gt_instances_from_semantic(gt_sem_hw, num_classes=num_classes, min_area=min_gt_area)
        n_gt = int(tgt["labels"].numel())

        keep = scores[b] >= float(score_thresh)

        if keep.any():
            kept_idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
            if kept_idx.numel() > int(topk_preds):
                topk = torch.topk(scores[b, kept_idx], k=int(topk_preds), dim=0).indices
                kept_idx = kept_idx[topk]
        else:
            kept_idx = torch.empty((0,), dtype=torch.long, device=pred_logits_bqc.device)

        pred_logits = pred_logits_bqc[b, kept_idx]
        pred_masks = pred_masks_bqhw[b, kept_idx]
        n_pred = int(pred_logits.shape[0])

        totals["n_gt"] += n_gt
        totals["n_pred"] += n_pred

        if n_gt == 0 or n_pred == 0:
            continue

        outputs_m = {
            "pred_logits": pred_logits.unsqueeze(0),
            "pred_masks": pred_masks.unsqueeze(0),
        }
        targets_m = [{"labels": tgt["labels"], "masks": tgt["masks"]}]

        indices = matcher(outputs_m, targets_m)[0]
        src_idx, tgt_idx = indices
        if src_idx.numel() == 0:
            continue

        totals["n_match"] += int(src_idx.numel())

        pred_lab = pred_labels[b, kept_idx][src_idx]
        gt_lab = tgt["labels"][tgt_idx]
        totals["class_correct"] += int((pred_lab == gt_lab).sum().item())

        pm = (pred_masks[src_idx].sigmoid() > float(mask_thresh))
        gm = (tgt["masks"][tgt_idx] > 0.5)

        inter = (pm & gm).flatten(1).sum(1).float()
        union = (pm | gm).flatten(1).sum(1).float().clamp_min(1.0)
        iou = inter / union
        dice = (2.0 * inter) / (pm.flatten(1).sum(1).float() + gm.flatten(1).sum(1).float()).clamp_min(1.0)

        totals["iou_sum"] += float(iou.sum().item())
        totals["dice_sum"] += float(dice.sum().item())

    return totals

def dice_coeff(pred, target, num_cls=7, epsilon=1e-6):

    assert pred.shape == target.shape, "pred/target 形状不一致"
    assert target.max().item() <= num_cls and target.min().item() >= 0, "Target 越界"
    assert pred.max().item()   <= num_cls and pred.min().item()   >= 0, "Pred 越界"

    C = num_cls + 1

    pred_one_hot   = F.one_hot(pred.long(),   num_classes=C).permute(0,3,1,2).float()
    target_one_hot = F.one_hot(target.long(), num_classes=C).permute(0,3,1,2).float()

    dims = (0, 2, 3)
    intersection = torch.sum(pred_one_hot * target_one_hot, dim=dims)
    union        = torch.sum(pred_one_hot, dim=dims) + torch.sum(target_one_hot, dim=dims)
    dice_scores  = (2.0 * intersection + epsilon) / (union + epsilon)

    foreground = slice(1, None)
    present = (torch.sum(target_one_hot, dim=dims) > 0)
    present_fg = present[foreground]
    dice_fg = dice_scores[foreground]

    if present_fg.any():
        mean_dice = (dice_fg * present_fg.float()).sum() / (present_fg.float().sum())
    else:
        mean_dice = torch.tensor(0.0, device=dice_fg.device)

    return mean_dice

class MaskFormer_track():
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

        self.model = MaskFormerModel_track(cfg)
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

        run_name = datetime.datetime.now().strftime("swin-%Y-%m-%d-%H-%M")

    def _reset_classwise_memory(self):

        model = self.model

        if hasattr(model, "module"):
            model = model.module

        if hasattr(model, "classwise_memory"):
            model.classwise_memory.reset_memory()

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

    def load_model(self, pretrain_weights):
        try:
            state_dict = torch.load(pretrain_weights, map_location='cuda:0')
            ckpt_dict = state_dict['model']
        except:
            with open(pretrain_weights, "rb") as f:
                obj = pickle.load(f)
            ckpt_dict = obj.get("model", obj)

        dict_new={}
        print(self.model.state_dict().keys())

        print(ckpt_dict.keys())
        print('loaded pretrained weights form %s !' % pretrain_weights)
        if self.cfg.bina:
            for k, v in ckpt_dict.items():

                if k.startswith('module.'):

                    dict_new[k[7:]] = v

                else:
                    dict_new[k] = v
        else:
            for k, v in ckpt_dict.items():
                if k.startswith('module.'):
                        dict_new[k[7:]] = v
                else:
                    dict_new[k] = v

        self.last_lr = 4e-5

        self.start_epoch = 70
        self.model = load_parallal_model(self.model, dict_new)

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

        self.last_lr = 6e-5

        self.start_epoch = 0

        self.model.load_state_dict(model_dict, strict=False)

    def _training_init(self, cfg):

        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION

        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        boundary_weight = cfg.MODEL.MASK_FORMER.BOUNDARY_WEIGHT
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
        TrainPath=[]
        ValPath=[]

        max_score = 0.5

        version = 0
        if self.cfg.dataset == 'EndoVis2017':
            train_dataloader, val_dataloader = dataloaders.get_dataloaders(
                train_paths_temp,test_paths_temp, self.cfg,batch_size=self.cfg.batch_size,num_workers=self.cfg.workers,version=version,bina=self.cfg.bina,)
        elif self.cfg.dataset == 'EndoVis2018':
            train_dataloader, val_dataloader = dataloaders.get_dataloaders(
                train_paths_temp, test_paths_temp, self.cfg,batch_size=self.cfg.batch_size,num_workers=self.cfg.workers,version=version,bina=self.cfg.bina,)

        if self.cfg.inferonly:
            score = self.evaluate(val_dataloader, 0, writer)
            print(f"Evaluation score: {score}")
            self.summary_writer.close()
            return

        for epoch in range(self.start_epoch + 1, n_epochs + 1):
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
        per_class_train={}
        end = time.time()
        self.istrain = True

        for i, (data, target,name) in enumerate(data_loader):
            train_iou={}
            if self.cfg.bina:
                inputs_left=data[0].to(device=self.device, non_blocking=True)
                inputs_right= data[1].to(device=self.device, non_blocking=True)
                flow_l2r=data[2].to(device=self.device, non_blocking=True)
                target = target.to(device=self.device, non_blocking=True)

            else:

                data_time = time.time() - end
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
    @torch.no_grad()
    def evaluate(self, eval_loader, epoch, writer):

        self.model.eval()

        dice_score = []

        losses_list_val = []
        loss_ce_list_val = []
        loss_dice_list_val = []
        loss_mask_list_val = []

        preds_list = []
        targets_list = []
        per_class_val={}

        val_pred = torch.zeros((len(eval_loader.dataset), 128, 128), dtype=torch.long, device=self.device)
        val_target = torch.zeros((len(eval_loader.dataset), 128, 128), dtype=torch.long, device=self.device)
        val_image = torch.zeros((len(eval_loader.dataset), 128, 128, 3), device=self.device)
        names=[]

        self.istrain = False
        with torch.no_grad():

            for i, (data, target,name) in enumerate(eval_loader):
                if self.cfg.bina:
                    inputs_left=data[0].to(device=self.device, non_blocking=True)
                    inputs_right= data[1].to(device=self.device, non_blocking=True)
                    flow_l2r=data[2].to(device=self.device, non_blocking=True)
                    gt_mask = target.to(device=self.device, non_blocking=True)

                else:
                    inputs= data.to(device=self.device, non_blocking=True)
                    gt_mask = target.to(device=self.device, non_blocking=True)

                    name_idx = torch.arange(len(name), device=self.device)
                    outputs,losses= self.model(inputs,target,self.istrain,name_idx, epoch, i, name)

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
                losses_list_val.append(loss.mean().item())
                loss_ce_list_val.append(loss_ce.mean().item())
                loss_dice_list_val.append(loss_dice.mean().item())
                loss_mask_list_val.append(loss_mask.mean().item())

                outputs = self.post_processing(outputs)
                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]

                start = int(i * eval_loader.batch_size)
                if self.cfg.inferonly:

                    inputs = F.interpolate(inputs[:,-1], size=(128, 128), mode="bilinear", align_corners=False)
                    x_denorm = denorm_bchw(inputs,  [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]).clamp(0, 1)*255
                    pred_masks_val = self.semantic_inference(mask_cls_results, mask_pred_results)

                else:
                    pred_masks_val = self.semantic_inference(mask_cls_results, mask_pred_results)

                bs = int(pred_masks_val.shape[0])
                if self.cfg.inferonly:
                    val_image[start:start+bs, :, :, :] = x_denorm.permute(0,2,3,1)

                val_pred[start:start+bs, :, :] = pred_masks_val.argmax(dim=1).to(torch.long)
                val_target[start:start+bs, :, :] = gt_mask[:, -1, :, :].to(torch.long)
                names += name
            preds_all = val_pred
            targets_all = val_target
            if self.cfg.inferonly:

                iou_mean, dice_mean = compute_iou_and_dice(val_pred, val_target)
                print("IoU mean:", iou_mean.item())
                print("Dice mean:", dice_mean.item())
                preds_all = val_pred.long()
                targets_all = val_target.long()

                num_classes = 7
                iou_thresh = float(getattr(self.cfg, 'EVAL_IOU_THRESH', 0.8))
                eps = 1e-6

                _, _, dice_bin = self._eval_binary_fg_and_dice(preds_all, targets_all, eps=eps)

                iou_mat, per_class_cm, per_class_auc = self._eval_image_level_per_class_tp_tn_fp_fn(
                    preds_all=preds_all,
                    targets_all=targets_all,
                    num_classes=num_classes,
                    iou_thresh=iou_thresh,
                    eps=eps,
                )

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

                for cid in range(1, num_classes + 1):
                    auc_c = per_class_auc[cid]
                    if not (auc_c != auc_c):
                        writer.add_scalar(f'val/class_{cid}_auc', auc_c, epoch)

                save_dir = os.path.join(self.save_folder, 'eval_metrics')
                os.makedirs(save_dir, exist_ok=True)

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
            val_iou = eval_endovis(preds_all, targets_all, num_classes=7, ignore_background=True, device=self.device)
            if self.cfg.inferonly:
                for i in range(len(preds_all)):
                    dice_mean_new=dice_coeff(preds_all[i,:,:].unsqueeze(0), targets_all[i,:,:].unsqueeze(0).to(self.device))
                    dice_all.append(dice_mean_new.detach().cpu().numpy())

                    visualization_dir = os.path.join(self.cfg.output_dir, "visualizations")
                    overlay(preds_all[i,:,:], targets_all[i,:,:].to(self.device),names[i],dice_mean_new,visualization_dir,val_image[i])
                dice=np.mean(dice_all)
            score=val_iou['IoU']
            print(val_iou)

            writer.add_scalar('val/val_loss', np.mean(losses_list_val), epoch)
            writer.add_scalar('val/val_dice_loss', np.mean(loss_dice_list_val), epoch)
            writer.add_scalar('val/val_bce_loss', np.mean(loss_ce_list_val), epoch)
            writer.add_scalar('val/val_challengIoU', val_iou['challengIoU'], epoch)
            writer.add_scalar('val/val_IoU', val_iou['IoU'], epoch)
            writer.add_scalar('val/val_mcIoU', val_iou['mcIoU'], epoch)
            writer.add_scalar('val/val_mIoU', val_iou['mIoU'], epoch)
            for i in range(len(val_iou['cIoU_per_class'])):
                per_class_val['cIoU_per_class_'+str(i+1)]=torch.tensor(val_iou['cIoU_per_class'][i])
            writer.add_scalars('val/val_cIoU_per_class', per_class_val, epoch)

            print('evaluate dice: {0},loses: {1},/n'.format(score,np.mean(losses_list_val)))
            os.makedirs(self.cfg.TRAIN.LOG_DIR, exist_ok=True)
            log_path = os.path.join(self.cfg.TRAIN.LOG_DIR, f"{self.cfg.task}.txt")
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write('evaluate IIoU: {0},losses:{1}/n'.format(score,np.mean(losses_list_val)))
                progress = f'loss:{(np.mean(losses_list_val)):.6f} loss_ce:{(np.mean(loss_ce_list_val)):.6f} loss_dice:{(np.mean(loss_dice_list_val)):.6f} loss_mask:{(np.mean(loss_mask_list_val)):.6f}\n '
                f.write(progress)
            del val_pred,val_target

        return score

    def post_processing(self, outputs, aux_logits=None):

        out_logits = outputs['pred_logits'][:,-1,:,:]

        if aux_logits is not None:
            aux_logits = aux_logits[0]
            aux_logits = torch.mean(aux_logits, dim=0)
        outputs['pred_logits'] = out_logits
        outputs['ids'] = [torch.arange(0, outputs['pred_masks'].size(2))]
        outputs['pred_masks'] =outputs['pred_masks'][:,-1,:,:]

        if aux_logits is not None:
            return outputs, aux_logits
        return outputs

    def _get_dice(self, predict, target):
        smooth = 1e-5

        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1)
        den = predict.sum(-1) + target.sum(-1)
        score = (2 * num + smooth).sum(-1) / (den + smooth).sum(-1)
        return score.mean()

    def _get_binary_mask(self, target):

        num,y, x = target.size()
        target_onehot = torch.zeros(num,self.num_classes + 1, y, x).to(target.device)
        target_onehot = target_onehot.scatter(dim=1, index=target.unsqueeze(1), value=1)
        return target_onehot

    def _get_binary_mask_one(self, target):
        target=target.to(torch.int64)
        y, x = target.size()
        target_onehot = torch.zeros(self.num_classes + 1, y, x).to(target.device)
        target_onehot = target_onehot.scatter(dim=0, index=target.unsqueeze(0), value=1)
        return target_onehot[1:-1]

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[...,:-1]

        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
        semseg= F.softmax(semseg, dim=1)
        return semseg

    def semantic_inference_topk(self, mask_cls, mask_pred,topk=2):

        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        mask_pred=nn.Threshold(0.5,0)(mask_pred)
        B, Q, H, W = mask_pred.shape
        C = mask_cls.shape[-1]

        max_values, max_indices = torch.max(mask_cls, dim=2, keepdim=True)
        mask_cls_sparse = torch.zeros_like(mask_cls)
        mask_cls_sparse.scatter_(2, max_indices, max_values)

        mask_cls_transposed = mask_cls_sparse.permute(0, 2, 1)
        topk_values, topk_indices = torch.topk(mask_cls_transposed, k=topk, dim=2)

        valid_mask = topk_values > 0

        mask_pred_all=[]
        semseg_all=[]
        for b in range(B):

            print(valid_mask[b])
            topk_index=[topk_indices[b].flatten()[i] for i in range(len(topk_indices[b].flatten())) if valid_mask[b].flatten()[i]]
            topk_index=torch.tensor(topk_index)
            topk_indices_batch=torch.unique(topk_index).to(mask_cls.device)

            mask_pred_batch= torch.index_select(mask_pred[b], 0, topk_indices_batch)
            mask_cls_batch= torch.index_select(mask_cls[b], 0, topk_indices_batch)

            semseg_batch = torch.einsum("qc,qhw->chw", mask_cls_batch, mask_pred_batch)
            semseg_all.append(semseg_batch)
        semseg=torch.stack(semseg_all)
        return semseg

    def _eval_binary_fg_and_dice(self, preds_all, targets_all, eps: float = 1e-6):

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

            pred_only = (p & (~g)).flatten(1).sum(1).float()
            gt_only = (g & (~p)).flatten(1).sum(1).float()

            tp = (union > 0) & (iou > iou_thresh)
            tn = (union == 0)
            undecided = (~tp) & (~tn)
            fp = undecided & (pred_only > gt_only)
            fn = undecided & (~(pred_only > gt_only))

            tn_c = int(tn.sum().item())
            fp_c = int(fp.sum().item())
            fn_c = int(fn.sum().item())
            tp_c = int(tp.sum().item())
            per_class_cm[cid] = np.array([[tn_c, fp_c], [fn_c, tp_c]], dtype=np.int64)

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

if __name__ == '__main__':
    def _get_binary_mask( num_classes,target):

        num,y, x = target.size()
        target_onehot = torch.zeros(num,num_classes + 1, y, x).to(target.device)
        target_onehot = target_onehot.scatter(dim=1, index=target.unsqueeze(1), value=1)
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
