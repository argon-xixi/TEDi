

import math
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score
from torch import nn
from torch.nn import functional as F

from .mIOU_new import eval_endovis
from .utils import compute_iou_and_dice, overlay


def denorm_bchw(x, mean, std):
    """将 [B, C, H, W] 图像按指定均值和标准差反归一化。"""
    mean = torch.as_tensor(mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    std = torch.as_tensor(std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return x * std + mean


def get_segimg(semseg): #加上背景后argmax
    # 输入为pred_mask (bn,7,h,w)
    background_prob = 1 - semseg.sum(dim=1, keepdim=True)
    semseg_with_bg = torch.cat([background_prob, semseg], dim=1)
    pred_mask=semseg_with_bg
    # pred_mask = semseg_with_bg.argmax(dim=1)
    return pred_mask #输出为(bn,h,w)





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


class EvaluationMixin:
    @torch.no_grad()
    def evaluate(self, eval_loader, epoch, writer):
        self.model.eval()
       
        losses_list_val = []
        loss_ce_list_val = []
        loss_dice_list_val = []
        loss_mask_list_val = []
        per_class_val={}
        val_pred = torch.zeros((len(eval_loader.dataset), 128, 128), dtype=torch.long, device=self.device)
        val_target = torch.zeros((len(eval_loader.dataset), 128, 128), dtype=torch.long, device=self.device)
        val_image = torch.zeros((len(eval_loader.dataset), 128, 128, 3), device=self.device)
        names=[]

        self.istrain = False
        with torch.no_grad():

            for i, (data, target,name) in enumerate(eval_loader):
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
    
            dice_all=[]
            val_iou = eval_endovis(preds_all, targets_all, num_classes=7, ignore_background=True, device=self.device)
            if self.cfg.inferonly:
                for i in range(len(preds_all)):
                    dice_mean_new=dice_coeff(preds_all[i,:,:].unsqueeze(0), targets_all[i,:,:].unsqueeze(0).to(self.device))
                    dice_all.append(dice_mean_new.detach().cpu().numpy())
                    overlay(preds_all[i,:,:], targets_all[i,:,:].to(self.device),names[i],dice_mean_new,'final_vis',val_image[i])
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
        """
        average the class logits and append query ids
        """

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

