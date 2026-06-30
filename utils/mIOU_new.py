import numpy as np
import torch

def eval_endovis(pred_masks, gt_masks, num_classes: int = 7,
                 ignore_background: bool = True, device: str = "cuda"):

    if isinstance(device, str):
        device = torch.device(device)

    pred_masks = pred_masks.to(device=device, dtype=torch.long)
    gt_masks = gt_masks.to(device=device, dtype=torch.long)

    B, H, W = pred_masks.shape

    class_ids = torch.arange(1, num_classes + 1, device=device).view(1, num_classes, 1, 1)

    pred_exp = pred_masks.view(B, 1, H, W)
    gt_exp = gt_masks.view(B, 1, H, W)

    pred_per_class = (pred_exp == class_ids).float()
    gt_per_class = (gt_exp == class_ids).float()

    intersection_all = (pred_per_class * gt_per_class).sum(dim=(0, 2, 3))
    union_all = pred_per_class.sum(dim=(0, 2, 3)) + gt_per_class.sum(dim=(0, 2, 3)) - intersection_all

    eps = 1e-7

    cum_I = intersection_all.sum()
    cum_U = union_all.sum()
    final_im_iou = (cum_I / (cum_U + eps)).item()

    intersection_frame = (pred_per_class * gt_per_class).sum(dim=(2, 3))
    union_frame = pred_per_class.sum(dim=(2, 3)) + gt_per_class.sum(dim=(2, 3)) - intersection_frame
    iou_frame = intersection_frame / (union_frame + eps)

    valid_frame_class = union_frame > 0

    gt_present_frame_class = gt_per_class.sum(dim=(2, 3)) > 0

    all_im_iou_acc = []
    all_im_iou_acc_challenge = []
    for b in range(B):
        valid_cls = valid_frame_class[b]
        if valid_cls.any():
            all_im_iou_acc.append(iou_frame[b][valid_cls].mean().item())

        chal_cls = gt_present_frame_class[b]
        if chal_cls.any():
            all_im_iou_acc_challenge.append(iou_frame[b][chal_cls].mean().item())

    mean_im_iou = float(np.mean(all_im_iou_acc)) if len(all_im_iou_acc) > 0 else 0.0
    mean_im_iou_challenge = float(np.mean(all_im_iou_acc_challenge)) if len(all_im_iou_acc_challenge) > 0 else 0.0

    class_ious = {c: [] for c in range(1, num_classes + 1)}
    for c in range(num_classes):
        cls_valid = valid_frame_class[:, c]
        if cls_valid.any():
            class_ious[c + 1] = iou_frame[cls_valid, c].detach().cpu().tolist()
        else:
            class_ious[c + 1] = []

    final_class_im_iou = torch.zeros(num_classes, dtype=torch.float32)
    cIoU_per_class = []
    for c in range(1, num_classes + 1):
        values = class_ious[c]
        if len(values) == 0:
            mean_c = 0.0
        else:
            mean_c = float(np.mean(values))
        final_class_im_iou[c - 1] = mean_c
        cIoU_per_class.append(round(mean_c * 100.0, 3))

    non_empty = [v for v in class_ious.values() if len(v) > 0]
    if len(non_empty) > 0:
        mean_class_iou = float(np.mean([np.mean(v) for v in non_empty]))
    else:
        mean_class_iou = 0.0

    endovis_results = {
        "challengIoU": round(mean_im_iou_challenge * 100.0, 3),

        "IoU": round(mean_im_iou * 100.0, 3),
        "mcIoU": round(mean_class_iou * 100.0, 3),

        "mIoU": round(final_im_iou * 100.0, 3),
        "cIoU_per_class": cIoU_per_class,
    }

    return endovis_results

def compute_mask_IU_endovis(masks, target):

    assert target.shape[-2:] == masks.shape[-2:]
    temp = masks * target
    intersection = temp.sum()
    union = ((masks + target) - temp).sum()
    return intersection, union

def eval_endovis_bina(pred_masks, gt_masks, num_classes=7, ignore_background=True, device='cuda'):
    B,C,H, W = pred_masks.shape
    pred_masks=pred_masks.detach().cpu()
    gt_masks=gt_masks.detach().cpu()

    metrics = {
        "cum_I": torch.zeros(num_classes, device=device),
        "cum_U": torch.zeros(num_classes, device=device),
        "class_counts": torch.zeros(num_classes, device=device),
        "frame_iou": [],
        "frame_challenge_iou": []
    }

    for b in range(B):
        pred = pred_masks[b]
        gt = gt_masks[b]

        ks = torch.arange(1, 8, device=gt.device).view(-1, 1, 1)

        gt_mask = (gt == ks).to(torch.int8)
        pred_mask=pred

        intersection = (pred_mask * gt_mask).sum(dim=(1,2))
        union = (pred_mask + gt_mask).sum(dim=(1,2)) - intersection

        present_classes = torch.unique(gt)
        present_classes = present_classes[present_classes > 0]
        valid_classes = present_classes[present_classes <= num_classes] - 1

        metrics["cum_I"] += intersection
        metrics["cum_U"] += union
        metrics["class_counts"] += (union > 0).float()

        frame_iou = intersection / torch.clamp(union, min=1e-7)
        challenge_iou = frame_iou[valid_classes.long()]

        if len(challenge_iou) > 0:
            metrics["frame_challenge_iou"].append(challenge_iou.mean().item())
        if len(present_classes) > 0:
            metrics["frame_iou"].append(frame_iou.mean().item())

    epsilon = 1e-7
    class_ious = metrics["cum_I"] / (metrics["cum_U"] + epsilon)
    class_weights = metrics["class_counts"] / (metrics["class_counts"].sum() + epsilon)

    return {
        "mIoU": (class_ious.mean() * 100).item(),
        "IoU": np.mean(metrics["frame_iou"]) * 100 if metrics["frame_iou"] else 0,
        "challengIoU": np.mean(metrics["frame_challenge_iou"]) * 100 if metrics["frame_challenge_iou"] else 0,
        "mcIoU": (class_ious[metrics["class_counts"] > 0].mean() * 100).item(),
        "cIoU_per_class": [round((iou * 100).item(), 3) for iou in class_ious]
    }

if __name__ == "__main__":
    x1=torch.tensor([[[1,1,1],[1,1,1],[1,1,1]]])
    x2= torch.tensor([[[1,1,1],[1,1,1],[1,1,1]]])
    print(eval_endovis(x1,x2,num_classes=2))