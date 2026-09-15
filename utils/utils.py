

import os

import cv2
import numpy as np
import torch

PALETTE_RGB = np.array(
    [
        [0, 0, 0],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [255, 255, 255],
    ],
    dtype=np.uint8,
)

def _as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)

def _apply_color_overlay(image_bgr, labels, alpha=0.6):
    labels = np.clip(labels, 0, len(PALETTE_RGB) - 1).astype(np.uint8)
    color_bgr = cv2.cvtColor(PALETTE_RGB[labels], cv2.COLOR_RGB2BGR)
    foreground = labels > 0
    output = image_bgr.copy()
    output[foreground] = cv2.addWeighted(
        image_bgr[foreground], 1.0 - alpha, color_bgr[foreground], alpha, 0
    )
    return output

def overlay(pred_mask, target, name, dice, task, img, alpha=0.6, out_size_hw=(256, 256), save=True):

    pred = np.squeeze(_as_numpy(pred_mask))
    image = _as_numpy(img)
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        if image.size and float(image.max()) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    if pred.shape != image.shape[:2]:
        pred = cv2.resize(pred.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    output = _apply_color_overlay(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), pred, alpha)
    output = cv2.resize(output, (int(out_size_hw[1]), int(out_size_hw[0])), interpolation=cv2.INTER_AREA)
    if save:
        output_dir = task
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, f"{name}.png"), output)
    return output

def compute_iou_and_dice(pred_mask, target_mask, num_classes=8, ignore_background=True, eps=1e-6):

    pred = pred_mask.long()
    target = target_mask.long()
    first_class = 1 if ignore_background else 0
    ious, dices = [], []
    for class_id in range(first_class, num_classes):
        pred_class = pred == class_id
        target_class = target == class_id
        union = (pred_class | target_class).sum().float()
        if union == 0:
            continue
        intersection = (pred_class & target_class).sum().float()
        total = pred_class.sum().float() + target_class.sum().float()
        ious.append((intersection + eps) / (union + eps))
        dices.append((2 * intersection + eps) / (total + eps))
    if not ious:
        zero = torch.tensor(0.0, device=pred.device)
        return zero, zero
    return torch.stack(ious).mean(), torch.stack(dices).mean()
