# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

from .point_features import point_sample



def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
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
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
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
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points
    #新版本，把不同类别的query分开
    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []
        new_matcher = False
        if new_matcher:

            for b in range(bs):
                out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes+1]
                out_mask = outputs["pred_masks"][b]

                tgt_ids = targets[b]["labels"]            # [num_total_targets]
                tgt_mask = targets[b]["masks"].to(out_mask)

                # ===== 原来的 cost 计算 =====
                cost_class = -out_prob[:, tgt_ids]         # [num_queries, num_total_targets]

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
                )  # [num_queries, num_total_targets]

                # ====== 在这里加“按类别划分 query 区域”的约束 ======
                # 假设 out_prob 的最后一维：num_classes_all = num_fg_classes + 1(no-object)
                num_classes_all = out_prob.shape[-1]
                num_fg_classes = num_classes_all - 1

                # 你这里 8 类，可以直接设为 8；也可以用 num_fg_classes 保持通用
                num_classes_for_split = 8  # 或者 = num_fg_classes

                queries_per_class = num_queries // num_classes_for_split
                extra = num_queries % num_classes_for_split

                BIG_COST = 1e6

                # C 的列顺序和 tgt_ids 一一对应
                # 针对每个 GT，查看它的类别，算出允许的 query 区间，把其它 query 位置加上大惩罚
                for j, cls_id in enumerate(tgt_ids.tolist()):
                    # 如果 0 是背景而你不想给它分区域，可以直接跳过或单独处理
                    # 比如：背景随便用所有 query
                    # if cls_id == 0:
                    #     continue

                    # 把类别 ID 映射到 0 ~ num_classes_for_split-1 的 slot
                    # 若 0 是背景、1~7 是前景，可以用：slot = cls_id - 1
                    slot = cls_id

                    if slot < 0 or slot >= num_classes_for_split:
                        # 不在我们要划分的这几类里，直接跳过或者放宽约束
                        continue

                    # 处理 100 不能被 8 整除的情况，前 extra 个类多分 1 个 query
                    start = slot * queries_per_class + min(slot, extra)
                    end = start + queries_per_class + (1 if slot < extra else 0)

                    # 对当前这个 GT（列 j），只允许 [start, end) 这一段的 query
                    # 其它 query 的代价加一个超级大的值
                    if start > 0:
                        C[:start, j] += BIG_COST
                    if end < num_queries:
                        C[end:, j] += BIG_COST

                # ====== 约束结束，继续原来的匈牙利匹配 ======
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
            """More memory-friendly matching"""
            bs, num_queries = outputs["pred_logits"].shape[:2]

            indices = []

            # Iterate through batch size
            for b in range(bs):
                out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes+1]
                out_mask = outputs["pred_masks"][b]  #(64,128,128) [num_queries, H_pred, W_pred]

                tgt_ids = targets[b]["labels"] # [1,2,3, ……]            
                tgt_mask = targets[b]["masks"].to(out_mask) #(1,512,512) [c, h, w] c = len(tgt_ids) 
                # Compute the classification cost. Contrary to the loss, we don't use the NLL,
                # but approximate it in 1 - proba[target class].
                # The 1 is a constant that doesn't change the matching, it can be ommitted.
                cost_class = -out_prob[:, tgt_ids] # [num_queries, num_total_targets]   #仅保留gt的类别，优化目标是概率，概率越高，成本越低
        

                #===========================Mask2Former方式====================================#
                # out_mask = out_mask[:, None] # [num_queries, 1, H_pred, W_pred]
                # tgt_mask = tgt_mask[:, None] # [c, 1, h, w]

                # # all masks share the same set of points for efficient matching!
                # point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
                # # get gt labels
                # tgt_mask = point_sample(
                #     tgt_mask, # [c, 1, h, w]
                #     point_coords.repeat(tgt_mask.shape[0], 1, 1), # [c, self.num_points, 2]
                #     align_corners=False,
                # ).squeeze(1) # [c, self.num_points]

                # out_mask = point_sample(
                #     out_mask,
                #     point_coords.repeat(out_mask.shape[0], 1, 1),
                #     align_corners=False,
                # ).squeeze(1) # [num_queries, self.num_points]            
                #===========================end====================================#

                #===========================MaskFormer方式====================================#
                # Flatten spatial dimension
                out_mask = out_mask.flatten(1)  # [num_queries, H*W]
                tgt_mask = tgt_mask.flatten(1)  # [num_total_targets, H*W]

                with autocast(enabled=False):
                # with autocast(device_type='cuda', enabled=False):
                    out_mask = out_mask.float()
                    tgt_mask = tgt_mask.float()
                    # Compute the focal loss between masks
                    cost_mask = batch_sigmoid_focal_loss(out_mask, tgt_mask)

                    # Compute the dice loss betwen masks
                    cost_dice = batch_dice_loss(out_mask, tgt_mask)
                # print(cost_class[23,:])
                # print(cost_dice[23,:])
                # print(cost_mask[23,:])
                # Final cost matrix
                C = (
                    self.cost_mask * cost_mask
                    + self.cost_class * cost_class
                    + self.cost_dice * cost_dice
                )
                # cost_mask和cost_dice占比远高于cost_class
                C = C.reshape(num_queries, -1).cpu() # [num_queries, num_total_targets]
                try:
                    indices.append(linear_sum_assignment(C)) #匈牙利匹配
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
            

        
        
    # @torch.no_grad()
    # def memory_efficient_forward(self, outputs, targets):
    #     """More memory-friendly matching"""
    #     device = outputs["pred_logits"].device
    #     t, num_queries = outputs["pred_logits"].shape[:2]
    #     num_total_targets = targets[0]["labels"].shape[0]
    #     cost_dice=torch.zeros((num_queries,num_total_targets)).to(device)
    #     cost_mask=torch.zeros((num_queries,num_total_targets)).to(device)
    #     cost_class=torch.zeros((num_queries,num_total_targets)).to(device)
    #     # Iterate through batch size
        
    #     for b in range(t):
    #         out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes+1]
    #         out_mask = outputs["pred_masks"][b]  #(64,128,128) [num_queries, H_pred, W_pred]

    #         tgt_ids = targets[b]["labels"] # [1,2,3, ……]            
    #         tgt_mask = targets[b]["masks"].to(out_mask) #(1,512,512) [c, h, w] c = len(tgt_ids) 
    #         # Compute the classification cost. Contrary to the loss, we don't use the NLL,
    #         # but approximate it in 1 - proba[target class].
    #         # The 1 is a constant that doesn't change the matching, it can be ommitted.
    #         cost_class += -out_prob[:, tgt_ids] # [num_queries, num_total_targets]   #仅保留gt的类别，优化目标是概率，概率越高，成本越低
    #         indices = []
    #         out_mask = out_mask.flatten(1)  # [num_queries, H*W]
    #         tgt_mask = tgt_mask.flatten(1)  # [num_total_targets, H*W]

    #         with autocast(enabled=False):
    #         # with autocast(device_type='cuda', enabled=False):
    #             out_mask = out_mask.float()
    #             tgt_mask = tgt_mask.float()
    #             # Compute the focal loss between masks
    #             cost_mask += batch_sigmoid_focal_loss(out_mask, tgt_mask)

    #             # Compute the dice loss betwen masks
    #             cost_dice += batch_dice_loss(out_mask, tgt_mask)

    #         #===========================Mask2Former方式====================================#
    #         # out_mask = out_mask[:, None] # [num_queries, 1, H_pred, W_pred]
    #         # tgt_mask = tgt_mask[:, None] # [c, 1, h, w]

    #         # # all masks share the same set of points for efficient matching!
    #         # point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
    #         # # get gt labels
    #         # tgt_mask = point_sample(
    #         #     tgt_mask, # [c, 1, h, w]
    #         #     point_coords.repeat(tgt_mask.shape[0], 1, 1), # [c, self.num_points, 2]
    #         #     align_corners=False,
    #         # ).squeeze(1) # [c, self.num_points]

    #         # out_mask = point_sample(
    #         #     out_mask,
    #         #     point_coords.repeat(out_mask.shape[0], 1, 1),
    #         #     align_corners=False,
    #         # ).squeeze(1) # [num_queries, self.num_points]            
    #         #===========================end====================================#

    #         #===========================MaskFormer方式====================================#
            
            
            
    #     # Final cost matrix
    #     C = (
    #         self.cost_mask * cost_mask
    #         + self.cost_class * cost_class
    #         + self.cost_dice * cost_dice
    #     )
    #     C = C.reshape(num_queries, -1).cpu() # [num_queries, num_total_targets]
    #     indices_restult = linear_sum_assignment(C)
    #     for i in range(t):
    #         indices.append(indices_restult) #匈牙利匹配
    #     # indices.append(linear_sum_assignment(C))

    #     return [
    #         (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
    #         for i, j in indices
    #     ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
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
