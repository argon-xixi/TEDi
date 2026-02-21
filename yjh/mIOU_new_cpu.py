import numpy as np 
import cv2 
import torch 
import os 
import os.path as osp 
import re 

def eval_endovis(pred_masks, gt_masks, num_classes=7, ignore_background=True, device='cuda'):
  
    # B,  H, W = pred_masks.shape
    
    # # assert gt_masks.shape == (B, H, W), "真值形状不匹配"
    
    # # 新版本
    
    
    # # 初始化统计量
    # metrics = {
    #     "cum_I": torch.zeros(num_classes, device=device),
    #     "cum_U": torch.zeros(num_classes, device=device),
    #     "class_counts": torch.zeros(num_classes, device=device),
    #     "frame_iou": [],
    #     "frame_challenge_iou": []
    # }
    
    # # 批量计算每个样本的交并比
    # for b in range(B):
    #     pred = pred_masks[b]  # (H, W)
    #     gt = gt_masks[b]      # (H, W)
        
    #     # 生成类别掩码矩阵 (num_classes, H, W)
    #     class_matrix = torch.arange(1, num_classes+1, device=device)[:, None, None]
        
    #     # 并行计算所有类别的IoU
    #     pred_mask = (pred == class_matrix).float()  # (C, H, W)
    #     gt_mask = (gt == class_matrix).float()      # (C, H, W)
        
    #     intersection = (pred_mask * gt_mask).sum(dim=(1,2))  # (C,)
    #     union = (pred_mask + gt_mask).sum(dim=(1,2)) - intersection
        
    #     # 统计有效类别
    #     present_classes = torch.unique(gt)
    #     present_classes = present_classes[present_classes > 0]
    #     valid_classes = present_classes[present_classes <= num_classes] - 1  # 转换为0-based索引
        
    #     # 更新全局统计
    #     metrics["cum_I"] += intersection
    #     metrics["cum_U"] += union
    #     metrics["class_counts"] += (union > 0).float()
        
    #     # 计算当前帧指标
    #     frame_iou = intersection / torch.clamp(union, min=1e-7)
    #     challenge_iou = frame_iou[valid_classes.long()] #作为整数索引
        
    #     if len(challenge_iou) > 0:
    #         metrics["frame_challenge_iou"].append(challenge_iou.mean().item())
    #     if len(present_classes) > 0:
    #         metrics["frame_iou"].append(frame_iou.mean().item())
    
    # # 计算最终指标
    # epsilon = 1e-7
    # class_ious = metrics["cum_I"] / (metrics["cum_U"] + epsilon)
    # class_weights = metrics["class_counts"] / (metrics["class_counts"].sum() + epsilon)
    
    # return {
    #     "mIoU": (class_ious.mean() * 100).item(),  # 全局像素级平均
    #     "IoU": np.mean(metrics["frame_iou"]) * 100 if metrics["frame_iou"] else 0,  # 帧平均
    #     "challengIoU": np.mean(metrics["frame_challenge_iou"]) * 100 if metrics["frame_challenge_iou"] else 0,
    #     "mcIoU": (class_ious[metrics["class_counts"] > 0].mean() * 100).item(),  # 有效类别平均
    #     "cIoU_per_class": [round((iou * 100).item(), 3) for iou in class_ious]
    # }
    
    """
      评估批量分割结果的指标 (兼容ISINet评估逻辑)
    
    Args:
        pred_masks (Tensor): 预测mask (B, H, W) C=num_classes (0为背景)
        gt_masks (Tensor): 真值mask (B, H, W) 值范围0~num_classes (0为背景)
        num_classes (int): 有效类别数（不含背景）
        ignore_background (bool): 是否排除背景类别
        device (str): 计算设备
        
    Returns:
        dict: 包含四项指标和各类别IoU的字典
    """

    # 老版本 ISINET
    
    endovis_results = dict()
    num_classes = 7
    
    all_im_iou_acc = []
    all_im_iou_acc_challenge = []
    cum_I, cum_U = 0, 0
    class_ious = {c: [] for c in range(1, num_classes+1)}
    
    B, H, W = pred_masks.shape
    for index in range(B):
        prediction = pred_masks[index].detach().cpu().numpy()
        full_mask = gt_masks[index].detach().cpu()
        
        
        im_iou = []
        im_iou_challenge = []
        target = full_mask.numpy()
        gt_classes = np.unique(target)
        gt_classes.sort()
        gt_classes = gt_classes[gt_classes > 0] 
        if np.sum(prediction) == 0:
            if target.sum() > 0: 
                all_im_iou_acc.append(0)
                all_im_iou_acc_challenge.append(0)
                for class_id in gt_classes:
                    class_ious[class_id].append(0)
            continue

        gt_classes = torch.unique(full_mask)
        # loop through all classes from 1 to num_classes 
        for class_id in range(1, num_classes + 1): 

            current_pred = (prediction == class_id).astype(np.float64)
            # current_pred =prediction[class_id-1].astype(np.float64)
            current_target = (full_mask.numpy() == class_id).astype(np.float64)

            if current_pred.astype(np.float64).sum() != 0 or current_target.astype(np.float64).sum() != 0:
                i, u = compute_mask_IU_endovis(current_pred, current_target)     
                im_iou.append(i/u)
                cum_I += i
                cum_U += u
                class_ious[class_id].append(i/u)
                if class_id in gt_classes:
                    im_iou_challenge.append(i/u)
        
        if len(im_iou) > 0:
            all_im_iou_acc.append(np.mean(im_iou))
        if len(im_iou_challenge) > 0:
            all_im_iou_acc_challenge.append(np.mean(im_iou_challenge))

    # calculate final metrics
    final_im_iou = cum_I / (cum_U + 1e-15)
    mean_im_iou = np.mean(all_im_iou_acc)
    mean_im_iou_challenge = np.mean(all_im_iou_acc_challenge)

    final_class_im_iou = torch.zeros(9)
    cIoU_per_class = []
    for c in range(1, num_classes + 1):
        final_class_im_iou[c-1] = torch.tensor(class_ious[c]).float().mean()
        cIoU_per_class.append(round((final_class_im_iou[c-1]*100).item(), 3))
        
    mean_class_iou = torch.tensor([torch.tensor(values).float().mean() for c, values in class_ious.items() if len(values) > 0]).mean().item()
    
    endovis_results["challengIoU"] = round(mean_im_iou_challenge*100,3)
    endovis_results["IoU"] = round(mean_im_iou*100,3) #等权地先平均类别、再平均图片
    endovis_results["mcIoU"] = round(mean_class_iou*100,3)
    endovis_results["mIoU"] = round(final_im_iou*100,3) #按像素数加权：大目标/大面积样本权重大，小目标权重小
    
    endovis_results["cIoU_per_class"] = cIoU_per_class
    
    return endovis_results

def compute_mask_IU_endovis(masks, target):
    """compute iou used for evaluation
    """
    assert target.shape[-2:] == masks.shape[-2:]
    temp = masks * target
    intersection = temp.sum()
    union = ((masks + target) - temp).sum()
    return intersection, union


def eval_endovis_bina(pred_masks, gt_masks, num_classes=7, ignore_background=True, device='cuda'):
    B,C,H, W = pred_masks.shape
    pred_masks=pred_masks.detach().cpu()
    gt_masks=gt_masks.detach().cpu()
    # assert gt_masks.shape == (B, H, W), "真值形状不匹配"
    
    # 新版本
    
    
    # 初始化统计量
    metrics = {
        "cum_I": torch.zeros(num_classes, device=device),
        "cum_U": torch.zeros(num_classes, device=device),
        "class_counts": torch.zeros(num_classes, device=device),
        "frame_iou": [],
        "frame_challenge_iou": []
    }
    
    # 批量计算每个样本的交并比
    for b in range(B):
        pred = pred_masks[b]  # (H, W)
        gt = gt_masks[b]     # (c,H, W)
        
        ks = torch.arange(1, 8, device=gt.device).view(-1, 1, 1)  # shape (7, 1, 1)
    
    # 广播比较并转换为 int8
        gt_mask = (gt == ks).to(torch.int8)  # 广播比较，生成形状 (7, H, W)
        pred_mask=pred
        
        intersection = (pred_mask * gt_mask).sum(dim=(1,2))  # (C,)
        union = (pred_mask + gt_mask).sum(dim=(1,2)) - intersection
        
        # 统计有效类别
        present_classes = torch.unique(gt)
        present_classes = present_classes[present_classes > 0]
        valid_classes = present_classes[present_classes <= num_classes] - 1  # 转换为0-based索引
        
        # 更新全局统计
        metrics["cum_I"] += intersection
        metrics["cum_U"] += union
        metrics["class_counts"] += (union > 0).float()
        
        # 计算当前帧指标
        frame_iou = intersection / torch.clamp(union, min=1e-7)
        challenge_iou = frame_iou[valid_classes.long()] #作为整数索引
        
        if len(challenge_iou) > 0:
            metrics["frame_challenge_iou"].append(challenge_iou.mean().item())
        if len(present_classes) > 0:
            metrics["frame_iou"].append(frame_iou.mean().item())
    
    # 计算最终指标
    epsilon = 1e-7
    class_ious = metrics["cum_I"] / (metrics["cum_U"] + epsilon)
    class_weights = metrics["class_counts"] / (metrics["class_counts"].sum() + epsilon)
    
    return {
        "mIoU": (class_ious.mean() * 100).item(),  # 全局像素级平均
        "IoU": np.mean(metrics["frame_iou"]) * 100 if metrics["frame_iou"] else 0,  # 帧平均
        "challengIoU": np.mean(metrics["frame_challenge_iou"]) * 100 if metrics["frame_challenge_iou"] else 0,
        "mcIoU": (class_ious[metrics["class_counts"] > 0].mean() * 100).item(),  # 有效类别平均
        "cIoU_per_class": [round((iou * 100).item(), 3) for iou in class_ious]
    }
    
    

if __name__ == "__main__":
    x1=torch.tensor([[[1,1,1],[1,1,1],[1,1,1]]])
    x2= torch.tensor([[[1,1,1],[1,1,1],[1,1,1]]])
    print(eval_endovis(x1,x2,num_classes=2))