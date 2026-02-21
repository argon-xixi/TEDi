import numpy as np 
import cv2 
import torch 
import os 
import os.path as osp 
import re 

def eval_endovis(endovis_masks, gt_endovis_masks):
    """Given the predicted masks and groundtruth annotations, predict the challenge IoU, IoU, mean class IoU, and the IoU for each class
        
      ** The evaluation code is taken from the official evaluation code of paper: ISINet: An Instance-Based Approach for Surgical Instrument Segmentation
      ** at https://github.com/BCV-Uniandes/ISINet
      
    Args:
        endovis_masks (dict): the dictionary containing the predicted mask for each frame 
        gt_endovis_masks (dict): the dictionary containing the groundtruth mask for each frame 

    Returns:
        dict: a dictionary containing the evaluation results for different metrics 

    
    # 模拟预测掩码
    endovis_masks = {
        "frame1": np.array([[1, 1, 0], [2, 2, 0], [0, 0, 0]]),
        "frame2": np.array([[1, 0, 0], [2, 0, 0], [3, 3, 3]])
    }

    # 模拟地面真值掩码
    gt_endovis_masks = {
        "frame1": np.array([[1, 1, 0], [2, 0, 0], [0, 0, 0]]),
        "frame2": np.array([[1, 0, 0], [2, 2, 0], [3, 3, 0]])
    }
    """

    endovis_results = dict()
    num_classes = 1
    
    all_im_iou_acc = []
    all_im_iou_acc_challenge = []
    cum_I, cum_U = 0, 0
    class_ious = {c: [] for c in range(1, num_classes+1)}
    
    for idx in range(len(endovis_masks)):
        prediction=endovis_masks[idx].cpu().numpy()
       
        full_mask = gt_endovis_masks[idx]
        
        im_iou = []
        im_iou_challenge = []
        target = full_mask.cpu().numpy()
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
    endovis_results["IoU"] = round(mean_im_iou*100,3)
    endovis_results["mcIoU"] = round(mean_class_iou*100,3)
    endovis_results["mIoU"] = round(final_im_iou*100,3)
    
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