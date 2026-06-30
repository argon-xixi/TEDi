

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

from .point_features import point_sample

def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):

    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss

def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):

    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum("nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw

def batch_sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):

    hw = inputs.shape[1]

    prob = inputs.sigmoid()
    focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    focal_neg = (prob ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )
    if alpha >= 0:
        focal_pos = focal_pos * alpha
        focal_neg = focal_neg * (1 - alpha)

    loss = torch.einsum("nc,mc->nm", focal_pos, targets) + torch.einsum("nc,mc->nm", focal_neg, (1 - targets))

    return loss / hw

class HungarianMatcher(nn.Module):

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 0):

        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []
        new_matcher = False
        if new_matcher:

            for b in range(bs):
                out_prob = outputs["pred_logits"][b].softmax(-1)
                out_mask = outputs["pred_masks"][b]

                tgt_ids = targets[b]["labels"]
                tgt_mask = targets[b]["masks"].to(out_mask)

                cost_class = -out_prob[:, tgt_ids]

                out_mask_flat = out_mask.flatten(1)
                tgt_mask_flat = tgt_mask.flatten(1)
                with autocast(enabled=False):
                    out_mask_flat = out_mask_flat.float()
                    tgt_mask_flat = tgt_mask_flat.float()
                    cost_mask = batch_sigmoid_focal_loss(out_mask_flat, tgt_mask_flat)
                    cost_dice = batch_dice_loss(out_mask_flat, tgt_mask_flat)

                C = (
                    self.cost_mask * cost_mask
                    + self.cost_class * cost_class
                    + self.cost_dice * cost_dice
                )

                num_classes_all = out_prob.shape[-1]
                num_fg_classes = num_classes_all - 1

                num_classes_for_split = 8

                queries_per_class = num_queries // num_classes_for_split
                extra = num_queries % num_classes_for_split

                BIG_COST = 1e6

                for j, cls_id in enumerate(tgt_ids.tolist()):

                    slot = cls_id

                    if slot < 0 or slot >= num_classes_for_split:

                        continue

                    start = slot * queries_per_class + min(slot, extra)
                    end = start + queries_per_class + (1 if slot < extra else 0)

                    if start > 0:
                        C[:start, j] += BIG_COST
                    if end < num_queries:
                        C[end:, j] += BIG_COST

                C = C.reshape(num_queries, -1).cpu()
                try:
                    indices.append(linear_sum_assignment(C))
                except:
                    print(C)
                    print(cost_mask)
                    print(cost_dice)
                    print(cost_class)
                    print("error")
                    exit()

            return [
                (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices
            ]

        else:
            bs, num_queries = outputs["pred_logits"].shape[:2]

            indices = []

            for b in range(bs):
                out_prob = outputs["pred_logits"][b].softmax(-1)
                out_mask = outputs["pred_masks"][b]

                tgt_ids = targets[b]["labels"]
                tgt_mask = targets[b]["masks"].to(out_mask)

                cost_class = -out_prob[:, tgt_ids]

                out_mask = out_mask.flatten(1)
                tgt_mask = tgt_mask.flatten(1)

                with autocast(enabled=False):

                    out_mask = out_mask.float()
                    tgt_mask = tgt_mask.float()

                    cost_mask = batch_sigmoid_focal_loss(out_mask, tgt_mask)

                    cost_dice = batch_dice_loss(out_mask, tgt_mask)

                C = (
                    self.cost_mask * cost_mask
                    + self.cost_class * cost_class
                    + self.cost_dice * cost_dice
                )

                C = C.reshape(num_queries, -1).cpu()
                try:
                    indices.append(linear_sum_assignment(C))
                except:
                    print(C)
                    print(cost_mask)
                    print(cost_dice)
                    print(cost_class)
                    print("error")
                    exit()

            return [
                (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices
            ]

    @torch.no_grad()
    def forward(self, outputs, targets):

        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
