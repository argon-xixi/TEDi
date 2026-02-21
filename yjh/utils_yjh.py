import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F
from typing import List, Dict, Union, Optional
import numpy as np
import os
import cv2
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

# def instance_inference_pure( 
#     mask_cls: torch.Tensor,                 # [Q,C+1] or [B,Q,C+1]，最后一列为 no-object
#     mask_pred: torch.Tensor,                # [Q,H,W] or [B,Q,H,W]，mask logits
#     pic_name,
#     test_topk_per_image: int = 10,         # 每张图最多保留多少实例（在 Q×C 上 topk）
#     panoptic_on: bool = False,              # 开启则仅保留 thing 类（暂未使用）
#     thing_class_ids: Optional[Union[List[int], set]] = None,  # 连续 id：0..C-1（暂未使用）
#     box_from_mask: bool = False,            # 是否由 mask 计算 bbox（否则置零）
#     mask_bin_thresh: float = 0.0,           # mask logit 的二值阈值（0≈prob>0.5）
# ) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:

#     batched = (mask_cls.dim() == 3)  # [B,Q,C+1]
#     if not batched:
#         mask_cls = mask_cls.unsqueeze(0)     # -> [1,Q,C+1]
#         mask_pred = mask_pred.unsqueeze(0)   # -> [1,Q,H,W]

#     B, Q, Cp1 = mask_cls.shape
#     C = Cp1 - 1
#     _, _, H, W = mask_pred.shape

#     outputs: List[Dict[str, torch.Tensor]] = []

#     for b in range(B):
#         mc = mask_cls[b]                      # [Q,C+1]
#         mp = mask_pred[b]                     # [Q,H,W]
#         device = mc.device

#         # 1) 分类得分（去掉 no-object）→ [Q,C]
#         scores_qc = F.softmax(mc, dim=-1)[:, 1:]   # [Q,C]

#         # 2) 先对每个 query 计算 mask 质量分 mask_scores_q [Q]
#         #    使用和原来完全一样的公式，只是对所有 Q 一次性算
#         pred_masks_bin_all = (mp > mask_bin_thresh).float()        # [Q,H,W]
#         mask_prob_all = mp.sigmoid()                               # [Q,H,W]

#         numer_all = (
#             mask_prob_all.flatten(1) * pred_masks_bin_all.flatten(1)
#         ).sum(1)                                                   # [Q]
#         denom_all = pred_masks_bin_all.flatten(1).sum(1).clamp_min(1e-6)  # [Q]
#         mask_scores_q = numer_all / denom_all                      # [Q]

#         # 3) 计算最终分数 final_scores_qc = 分类分 * mask 分
#         #    形状为 [Q,C]，再在这个上面做 topk
#         final_scores_qc = scores_qc * mask_scores_q.unsqueeze(1)   # [Q,C]

#         # 4) 在 (Q×C) 的 final_scores 上 topk
#         flat_final_scores = final_scores_qc.reshape(-1)            # [Q*C]
#         K = min(test_topk_per_image, flat_final_scores.numel())
#         if K == 0:
#             outputs.append({
#                 "image_size": (H, W),
#                 "pred_masks": torch.zeros((0, H, W), dtype=torch.float32, device=device),
#                 "pred_boxes": torch.zeros((0, 4),    dtype=torch.float32, device=device),
#                 "scores":     torch.zeros((0,),      dtype=torch.float32, device=device),
#                 "pred_classes": torch.zeros((0,),    dtype=torch.long,    device=device),
#             })
#             continue

#         final_scores, topk_idx = flat_final_scores.topk(K, sorted=True)  # [K]

#         # 还原 (q,c)
#         labels_per_image = (topk_idx % C).long()              # [K]
#         topk_queries     = (topk_idx // C).long()             # [K]
#         print(labels_per_image, topk_queries)

#         # 5) 取出对应 mask logits / bin mask
#         sel_mask_logits = mp[topk_queries]                    # [K,H,W]
#         pred_masks_bin = pred_masks_bin_all[topk_queries]     # [K,H,W]

#         N = sel_mask_logits.shape[0]
#         if N == 0:
#             outputs.append({
#                 "image_size": (H, W),
#                 "pred_masks": torch.zeros((0, H, W), dtype=torch.float32, device=device),
#                 "pred_boxes": torch.zeros((0, 4),    dtype=torch.float32, device=device),
#                 "scores":     torch.zeros((0,),      dtype=torch.float32, device=device),
#                 "pred_classes": torch.zeros((0,),    dtype=torch.long,    device=device),
#             })
#             continue

#         # 6) bbox（保持原逻辑）
#         if box_from_mask:
#             pred_boxes = _masks_to_boxes(pred_masks_bin > 0.5)                # [N,4]
#         else:
#             pred_boxes = torch.zeros((N, 4), dtype=torch.float32, device=device)

#         outputs.append({
#             "image_size": (H, W),
#             "pred_masks": pred_masks_bin,
#             "pred_boxes": pred_boxes,
#             "scores": final_scores,              # 现在是基于 final_scores 排序后的 topk 分数
#             "pred_classes": labels_per_image,
#         })

#         # ====== 下面是你原来的可视化保存部分，逻辑保持不变，只是用新的 final_scores ======
#         topk_queries_np     = topk_queries.detach().cpu().numpy()
#         labels_per_image_np = labels_per_image.detach().cpu().numpy()
#         final_scores_np     = final_scores.detach().cpu().numpy()

#         # for i in range(min(64, len(topk_queries_np))):
#         #     mask = pred_masks_bin[i] * (labels_per_image_np[i] + 1)
#         #     mask = mask.detach().cpu().numpy().astype('uint8')
#         #     color_mask1 = palette[mask]
#         #     color_mask1 = cv2.cvtColor(color_mask1, cv2.COLOR_RGB2BGR)
#         #     save_dir = f'/data/yjh_data/Endovis2018/instance/2/{pic_name}'
#         #     if not os.path.exists(save_dir):
#         #         os.makedirs(save_dir)
#         #     cv2.imwrite(
#         #         f'{save_dir}/{topk_queries_np[i]}_{int(labels_per_image_np[i] + 1)}_{final_scores_np[i]}.png',
#         #         color_mask1
#         #     )

#     return outputs[0] if not batched else outputs


@torch.no_grad()
def instance_inference_pure(
    mode,
    mask_cls: torch.Tensor,                 # [Q,C+1] or [B,Q,C+1]，最后一列为 no-object
    mask_pred: torch.Tensor,                # [Q,H,W] or [B,Q,H,W]，mask logits
    pic_name,
    test_topk_per_image: int = 10,         # 每张图最多保留多少实例（在 Q×C 上 topk）
    panoptic_on: bool = False,              # 开启则仅保留 thing 类
    thing_class_ids: Optional[Union[List[int], set]] = None,  # 连续 id：0..C-1
    box_from_mask: bool = False,            # 是否由 mask 计算 bbox（否则置零）
    mask_bin_thresh: float = 0.0,           # mask logit 的二值阈值（0≈prob>0.5）
    draw: bool = False,
) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """
    返回（单图）：{
        'image_size': (H, W),
        'pred_masks': FloatTensor [N,H,W] (0/1),
        'pred_boxes': FloatTensor [N,4]  (xyxy),
        'scores':     FloatTensor [N],
        'pred_classes': LongTensor [N]  (0..C-1)
    }
    批量输入时返回上述字典的 list（长度为 B）。
    """
    mask_cls=mask_cls[:,:-1]
    # 统一成批量维
    batched = (mask_cls.dim() == 3)  # [B,Q,C+1]
    if not batched:
        mask_cls = mask_cls.unsqueeze(0)     # -> [1,Q,C+1]
        mask_pred = mask_pred.unsqueeze(0)   # -> [1,Q,H,W]

    B, Q, Cp1 = mask_cls.shape
    C = Cp1 - 1
    _, _, H, W = mask_pred.shape

    outputs: List[Dict[str, torch.Tensor]] = []
    
    for b in range(B):
        mc = mask_cls[b]                      # [Q,C+1]
        mp = mask_pred[b]                     # [Q,H,W]
        device = mc.device

        # 1) 分类得分（去掉 no-object）→ [Q,C]
        # scores_qc = F.softmax(mc, dim=-1)[:, 1:]   # [Q,C]
        scores_qc = F.softmax(mc, dim=-1)
        if mode == "topk":
            # 2) 在 (Q×C) 上 topk
            flat_scores = scores_qc.reshape(-1)                   # [Q*C]

            K = min(test_topk_per_image, flat_scores.numel())
            scores_per_image, topk_idx = flat_scores.topk(K, sorted=True)  # [K]
            

            # 还原 (q,c)
            labels_per_image = (topk_idx % C).long()              # [K]
            topk_queries     = (topk_idx // C).long()             # [K]
            # print(labels_per_image, topk_queries)
            # print(scores_per_image)
        
            # 3) 取出对应 mask logits
            sel_mask_logits = mp[topk_queries]                    # [K,H,W]
            N = sel_mask_logits.shape[0]
            if N == 0:
                outputs.append({
                    "image_size": (H, W),
                    "pred_masks": torch.zeros((0, H, W), dtype=torch.float32, device=device),
                    "pred_boxes": torch.zeros((0, 4),    dtype=torch.float32, device=device),
                    "scores":     torch.zeros((0,),      dtype=torch.float32, device=device),
                    "pred_classes": torch.zeros((0,),    dtype=torch.long,    device=device),
                })
                continue

            # 5) 二值化得到实例 mask（logit > thresh）
            pred_masks_bin = (sel_mask_logits > mask_bin_thresh).float()  # [N,H,W]

            # 6) 掩码质量分：mask 内平均概率
            mask_prob = sel_mask_logits.sigmoid()                          # [N,H,W]
            numer = (mask_prob.flatten(1) * pred_masks_bin.flatten(1)).sum(1)      # [N]
            denom = pred_masks_bin.flatten(1).sum(1).clamp_min(1e-6)               # [N]
            mask_scores = numer / denom                                           # [N]
            final_scores = scores_per_image * mask_scores                         # [N]
            
        elif mode == "all":    
            ################# 修改为仅输出类别query top1
            # scores_qc: [Q, C]
            scores_per_image, labels_per_image = torch.max(scores_qc, dim=1, keepdim=False)  # [Q]
            topk_queries = torch.arange(0, Q).long().to(device)

            # 3) 取出对应 mask logits
            sel_mask_logits = mp[topk_queries]  # [Q, H, W], 直接选择每个query对应的mask logits
            N = sel_mask_logits.shape[0]
            
            if N == 0:
                outputs.append({
                    "image_size": (H, W),
                    "pred_masks": torch.zeros((0, H, W), dtype=torch.float32, device=device),
                    "pred_boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
                    "scores": torch.zeros((0,), dtype=torch.float32, device=device),
                    "pred_classes": torch.zeros((0,), dtype=torch.long, device=device),
                })
                continue

            # 5) 二值化得到实例 mask（logit > thresh）
            pred_masks_bin = (sel_mask_logits > mask_bin_thresh).float()  # [N, H, W]

            # 6) 掩码质量分：mask 内平均概率
            mask_prob = sel_mask_logits.sigmoid()  # [N, H, W]
            numer = (mask_prob.flatten(1) * pred_masks_bin.flatten(1)).sum(1)  # [N]
            denom = pred_masks_bin.flatten(1).sum(1).clamp_min(1e-6)  # [N]
            mask_scores = numer / denom  # [N]
            final_scores = scores_per_image * mask_scores  # [N]
            
        # 7) bboxes
        if box_from_mask:
            pred_boxes = _masks_to_boxes(pred_masks_bin > 0.5)  # [N, 4]
        else:
            pred_boxes = torch.zeros((N, 4), dtype=torch.float32, device=device)

        outputs.append({
            "image_size": (H, W),
            "pred_masks": pred_masks_bin,
            "pred_boxes": pred_boxes,
            "scores": final_scores,
            "pred_classes": labels_per_image,
            "topk_queries": topk_queries,
        })

        topk_queries = topk_queries.detach().cpu().numpy()
        labels_per_image = labels_per_image.detach().cpu().numpy()
        final_scores = final_scores.detach().cpu().numpy()
        scores_per_image = scores_per_image.detach().cpu().numpy()
        mask_scores = mask_scores.detach().cpu().numpy()
        if draw:
            # 画出实例图
            for i in range(min(100, len(topk_queries))):
                mask = pred_masks_bin[i] * (labels_per_image[i] )
                mask = mask.detach().cpu().numpy().astype('uint8')
               
                color_mask1 = palette[mask]
                color_mask1 = cv2.cvtColor(color_mask1, cv2.COLOR_RGB2BGR)
                if not os.path.exists(f'/data1/yuanjiahong_files/bishe/EndoVis2018/instance/new_baseline_1/{pic_name}'):
                    os.makedirs(f'/data1/yuanjiahong_files/bishe/EndoVis2018/instance/new_baseline_1/{pic_name}')
                cv2.imwrite(f'/data1/yuanjiahong_files/bishe/EndoVis2018/instance/new_baseline_1/{pic_name}/{topk_queries[i]}_{int(labels_per_image[i] + 1)}_{scores_per_image[i]}_{mask_scores[i]}_{final_scores[i]}.png', color_mask1)

    return outputs[0] if not batched else outputs


import torch
from typing import List, Dict, Union

@torch.no_grad()
def topk_instances_to_semantic_overwrite(
    pic_name,
    per_image_result: Dict[str, torch.Tensor],
    num_classes: int,
    topk: int = 7,
    mask_thresh: float = 0.5,   # 用于将 mask 二值化；你的 pred_masks 已是0/1也没问题
    score_thresh: float = 0.5,  # 选出 Top-K 实例时，score 的下限
) -> torch.Tensor:
    """
    输入：单张图的实例结果（来自 instance_inference_pure 的单图字典）
    输出：语义分割 [H, W]，0=背景，1..num_classes=前景类别
    逻辑：先按 scores 选 Top-K 实例，然后“高分→低分，逐步覆盖（只在尚未填充处写入）”
    """
    pred_masks: torch.Tensor = per_image_result["pred_masks"]      # [N,H,W], float(0/1)
    scores:     torch.Tensor = per_image_result["scores"]          # [N]
    labels:     torch.Tensor = per_image_result["pred_classes"]    # [N], 0..C-1
    query_idx:  torch.Tensor= per_image_result["topk_queries"]                    # [N]
    N = pred_masks.shape[0]
    H, W = per_image_result["image_size"]

    device = pred_masks.device
    sem = torch.zeros((H, W), dtype=torch.long, device=device)     # 0=背景
    if N == 0:
        return sem
    # 1) 对score做threshold
    
    scores[scores <= score_thresh] = 0.0

    # 1) 选 Top-K（按分数从高到低）
    k = min(topk, N)
    order = torch.argsort(scores,descending=True)[:k]
    masks_k  = pred_masks[order]           # [k,H,W]
    labels_k = labels[order]               # [k]
    scores_k = scores[order]               # [k]
    query_idx_k = query_idx[order]         # [k]
    # scores_k = scores[order]             # 如需调试可保留

    # 2) 逐步覆盖：高分优先、仅在“未填充像素”处写入
    filled = torch.zeros((H, W), dtype=torch.bool, device=device)
    # mask=(masks_k[0] > 0.0).float()
    mask=masks_k[0].sigmoid()
    
    mask=cv2.applyColorMap((mask*255).cpu().numpy().astype('uint8'),cv2.COLORMAP_JET)
    # cv2.imwrite('/data/yjh_files/code/Mask2Former-Simplify-master/yjh/test/test_mask.png',mask)
    for i in range(k):
        if not scores_k[i]> 1e-3:
            continue
        # 将 mask 二值化（如果已是0/1，这一步等价于原样）

        
        bin_mask = (masks_k[i] >= mask_thresh)
        mask= bin_mask * (labels_k[i]+1)
        color_mask1 = palette[mask.detach().cpu().numpy().astype('uint8')]
        color_mask1=cv2.cvtColor(color_mask1,cv2.COLOR_RGB2BGR)
        # if not os.path.exists('/data/yjh_data/Endovis2018/instance/6//{}'.format(pic_name)):
        #     os.makedirs('/data/yjh_data/Endovis2018/instance/6/{}'.format(pic_name))
        # cv2.imwrite('/data/yjh_data/Endovis2018/instance/6/{}/{}_{}_{}.png'.format(pic_name,query_idx_k[i] , int(labels_k[i]+1),scores_k[i]),color_mask1)
        writable = bin_mask & (~filled)
       
        if writable.any():
            if writable.sum() > bin_mask.sum() * 0.5:
                # 类别+1 留出0作背景
                sem[writable] = int(labels_k[i]) + 1
                filled[writable] = True

    return sem


@torch.no_grad()
def batch_topk_instances_to_semantic_overwrite(
    picname,
    results: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]],
    num_classes: int,
    topk: int = 5,
    mask_thresh: float = 0.5,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    同时支持单图字典或批量 list 的输入；
    返回单图 [H,W] 或 list[[H,W], ...]
    """
    if isinstance(results, dict):
        return topk_instances_to_semantic_overwrite(picname,results, num_classes, topk, mask_thresh)
    else:
        return [topk_instances_to_semantic_overwrite(picname,r, num_classes, topk, mask_thresh) for r in results]


def _masks_to_boxes(bin_masks: torch.Tensor) -> torch.Tensor:
    """
    bin_masks: bool/float [N,H,W]；True/1 为前景。
    返回 FloatTensor [N,4] (x1,y1,x2,y2)。空掩码 → (0,0,0,0)。
    """
    if bin_masks.dtype != torch.bool:
        bin_masks = bin_masks > 0.5
    N, H, W = bin_masks.shape
    device = bin_masks.device
    boxes = torch.zeros((N, 4), dtype=torch.float32, device=device)
    for i in range(N):
        ys, xs = torch.where(bin_masks[i])
        if ys.numel() == 0:
            continue
        y1, y2 = ys.min().float(), ys.max().float()
        x1, x2 = xs.min().float(), xs.max().float()
        boxes[i] = torch.tensor([x1, y1, x2, y2], device=device)
    return boxes

def binary_mask_eval(mask_cls, mask_pred,target):
    cls_fg = F.softmax(mask_cls, dim=-1)[..., 1:]
    pred_binary_mask = mask_pred.sigmoid() > 0.5
    

def semantic_inference_with_bg(mask_cls, mask_pred, tau=0.5):
    # 前景类概率：[B,Q,C]（去掉no-object）
    # for i in range(mask_cls.shape[1]):
    #     print(F.softmax(mask_cls, dim=-1)[0][i])
    cls_fg = F.softmax(mask_cls, dim=-1)[..., 1:]
    scores, labels = cls_fg.max(dim=2) 
    print(labels[0])
    # 掩码概率：[B,Q,H,W]
    mask_p = mask_pred.sigmoid()
    # 每类前景分数图：[B,C,H,W]
    semseg = torch.einsum("bqc,bqhw->bchw", cls_fg, mask_p)

    # 每像素最大前景分数与其类别
    scores, labels = semseg.max(dim=1)          # scores:[B,H,W], labels:[B,H,W] in [0..C-1]
    # print(scores)
    # 阈值决定背景（0），否则类别+1（让0留给背景）
    pred = torch.where(scores > tau, labels + 1, torch.zeros_like(labels))
    return labels 

def visanddraw(pred_mask,target,name,dice,task,img):
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
        #     f'/data/yjh_files/code/Mask2Former-Simplify-master/test1/pred/{i}_{dice}.png',
        #     cv2.cvtColor(color_mask1, cv2.COLOR_RGB2BGR)
        # )
        # print('保存成功')

        # ================ 可视化真实标签 ================
        gt_mask_np = target.detach().cpu().numpy().astype(np.uint8)
        
        # 检查数值范围
        gt_mask_np = np.clip(gt_mask_np, 0, len(palette)-1)
        
        # 获取颜色映射
        color_mask2 = palette[gt_mask_np]
        
        
        img_flo = np.concatenate([color_mask1, color_mask2,img.detach().cpu().numpy()], axis=0)
        if os.path.exists(f'/data/yjh_files/code/Mask2Former-Simplify-master/{task}'):
            pass
        else:
            os.mkdir(f'/data/yjh_files/code/Mask2Former-Simplify-master/{task}')
        
        if os.path.exists(f'/data/yjh_files/code/Mask2Former-Simplify-master/{task}/dice>0.8'):
            pass
        else:
            os.mkdir(f'/data/yjh_files/code/Mask2Former-Simplify-master/{task}/dice>0.8')
        if os.path.exists(f'/data/yjh_files/code/Mask2Former-Simplify-master/{task}/dice<0.8'):
            pass
        else:
            os.mkdir(f'/data/yjh_files/code/Mask2Former-Simplify-master/{task}/dice<0.8')
        if os.path.exists(f'/data/yjh_files/code/Mask2Former-Simplify-master/{task}/dice<0.5'):
            pass
        else:
            os.mkdir(f'/data/yjh_files/code/Mask2Former-Simplify-master/{task}/dice<0.5')
        
        if dice >0.8:
            
            cv2.imwrite(
                f'/data/yjh_files/code/Mask2Former-Simplify-master/{task}/dice>0.8/{name}_{dice}.png',
                cv2.cvtColor(img_flo, cv2.COLOR_RGB2BGR)
            )
        if dice>0.5 and dice<=0.8:
            cv2.imwrite(
                f'/data/yjh_files/code/Mask2Former-Simplify-master/{task}/dice<0.8/{name}_{dice}.png',
                cv2.cvtColor(img_flo, cv2.COLOR_RGB2BGR)
            )
        if dice<=0.5:
            cv2.imwrite(
                f'/data/yjh_files/code/Mask2Former-Simplify-master/{task}/dice<0.5/{name}_{dice}.png',
                cv2.cvtColor(img_flo, cv2.COLOR_RGB2BGR)
            )
def binarize_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    将多类别 mask (0~7) 转为二值 mask：
    0 -> 0（背景）
    1~7 -> 1（前景）
    
    参数:
        mask: Tensor, 形状为 (B, H, W)，元素为 [0, 7] 的整数
    
    返回:
        bin_mask: Tensor, 形状为 (B, H, W)，元素为 {0., 1.} 的 float
    """
    # >0 的都视为前景
    bin_mask = (mask > 0).float()
    return bin_mask


def compute_iou_and_dice(pred_mask: torch.Tensor,
                         gt_mask: torch.Tensor,
                         eps: float = 1e-6):
    """
    计算二分类的 IoU 和 Dice（逐样本 + batch 平均）

    参数:
        pred_mask: 预测 mask, 形状 (B, H, W)，原始值为 0~7
        gt_mask:   GT mask,   形状 (B, H, W)，原始值为 0~7
        eps:       防止除零的极小值

    返回:
        iou_per_sample:  每个样本的 IoU，形状 (B,)
        dice_per_sample: 每个样本的 Dice，形状 (B,)
        iou_mean:        batch 平均 IoU，标量
        dice_mean:       batch 平均 Dice，标量
    """
    # 1. 转为二值（0 背景，1 前景）
    pred_bin = binarize_mask(pred_mask)
    gt_bin   = binarize_mask(gt_mask)

    # 2. 展开到 (B, N) 方便逐样本计算
    B = pred_bin.shape[0]
    pred_flat = pred_bin.view(B, -1)
    gt_flat   = gt_bin.view(B, -1)

    # 3. 计算交集与并集
    intersection = (pred_flat * gt_flat).sum(dim=1)          # (B,)
    pred_sum = pred_flat.sum(dim=1)                          # (B,)
    gt_sum   = gt_flat.sum(dim=1)                            # (B,)
    union = pred_sum + gt_sum - intersection                 # (B,)

    # 4. IoU 和 Dice
    iou_per_sample = (intersection + eps) / (union + eps)
    dice_per_sample = (2 * intersection + eps) / (pred_sum + gt_sum + eps)

    # 5. batch 平均
    iou_mean = iou_per_sample.mean()
    dice_mean = dice_per_sample.mean()

    return  iou_mean, dice_mean

import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 可选：更好的平滑（如果没装 scipy 也能跑，会自动降级）
try:
    from scipy.ndimage import gaussian_filter1d
    _HAS_SCIPY = True
except Exception:
    gaussian_filter1d = None
    _HAS_SCIPY = False


def _subsample_1d(x: torch.Tensor, max_n: int, seed: int = 0) -> torch.Tensor:
    x = x.flatten()
    if x.numel() <= max_n:
        return x
    g = torch.Generator(device=x.device)
    g.manual_seed(seed)
    idx = torch.randperm(x.numel(), generator=g, device=x.device)[:max_n]
    return x[idx]


def _smooth_density_from_hist(x_np: np.ndarray, bins=220, smooth_sigma=2.0, x_range=None):
    """hist(density=True) + 平滑 -> 密度曲线"""
    if x_np.size == 0:
        return None, None
    hist, edges = np.histogram(x_np, bins=bins, range=x_range, density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0

    if smooth_sigma and smooth_sigma > 0:
        if _HAS_SCIPY:
            hist = gaussian_filter1d(hist.astype(np.float64), sigma=smooth_sigma)
        else:
            # 降级：滑动平均
            k = int(max(3, round(smooth_sigma * 3)))
            if k % 2 == 0:
                k += 1
            pad = k // 2
            hpad = np.pad(hist, (pad, pad), mode="edge")
            kernel = np.ones(k, dtype=np.float64) / k
            hist = np.convolve(hpad, kernel, mode="valid")
    return centers, hist


@torch.no_grad()
def class_pixel_score_from_queries(mask_cls_logits, mask_pred_logits, class_id: int, mode="prob"):
    """
    输入:
      mask_cls_logits: [B, Q, C+1]  (最后一维最后一个通常是 no-object，不是数据集背景)
      mask_pred_logits: [B, Q, H, W]
    输出:
      score_c: [B, H, W]  该类别在每个像素的 score

    mode:
      - "prob": 用 softmax(mask_cls)*sigmoid(mask_pred)（最贴近你现在的 semseg）
      - "raw":  用 mask_cls_logits * mask_pred_logits（更像“logit空间”的 raw score）
      - "log_prob": 对 prob 结果取 log(score+eps)（更像 logits 的可视化尺度）
    """
    assert mask_cls_logits.ndim == 3 and mask_pred_logits.ndim == 4
    B, Q, Cp1 = mask_cls_logits.shape
    assert mask_pred_logits.shape[0] == B and mask_pred_logits.shape[1] == Q

    if mode == "prob" or mode == "log_prob":
        cls_prob = F.softmax(mask_cls_logits, dim=-1)[..., :-1]  # [B,Q,C] 去掉 no-object
        w = cls_prob[:, :, class_id]                             # [B,Q]
        m = mask_pred_logits.sigmoid()                           # [B,Q,H,W]
        score = torch.einsum("bq,bqhw->bhw", w, m)               # [B,H,W]
        if mode == "log_prob":
            score = torch.log(score.clamp_min(1e-12))
        return score

    elif mode == "raw":
        # 注意：这是一个“raw score”，不等于严格概率，但更适合做“logits分离度”诊断
        # [B,Q,C] 去掉 no-object
        w = mask_cls_logits[:, :, class_id]                     # [B,Q]
        m = mask_pred_logits.sigmoid()                                     # [B,Q,H,W] 直接用 pre-sigmoid
        score = torch.einsum("bq,bqhw->bhw", w, m)               # [B,H,W]
        return score
    else:
        raise ValueError(f"Unknown mode={mode}")


@torch.no_grad()
def extract_pos_neg_values(score_bhw, gt_bhw, class_id, ignore_index=255,
                           max_pos=60000, max_neg=60000, seed=0):
    """从 [B,H,W] score 中按 GT 条件抽取正/负样本的一维数组"""
    assert score_bhw.shape == gt_bhw.shape
    valid = (gt_bhw != ignore_index)
    pos = score_bhw[(gt_bhw == class_id) & valid]
    neg = score_bhw[(gt_bhw != class_id) & valid]
    pos = _subsample_1d(pos, max_pos, seed=seed + class_id * 13 + 1).cpu()
    neg = _subsample_1d(neg, max_neg, seed=seed + class_id * 13 + 2).cpu()
    return pos.numpy(), neg.numpy()


def plot_query_semseg_distribution(
    gt,                         # [B,H,W]
    class_ids,                  # list[int]
    class_names=None,           # list[str]
    base=None,                  # tuple(mask_cls_logits, mask_pred_logits) 或 None
    ours=None,                  # tuple(mask_cls_logits, mask_pred_logits) 或 None
    ignore_index=255,
    mode="prob",                # "prob" | "raw" | "log_prob"
    bins=220,
    smooth_sigma=2.0,
    max_pos=60000,
    max_neg=60000,
    seed=0,
    share_x=True,
    figsize_per_cell=(5.2, 3.8),
    title=None,
    save_path=None
):
    """
    画分布图：
      - 绿色: GT=class
      - 红色: GT≠class
      - baseline: 虚线
      - ours: 实线
    """
    assert base is not None or ours is not None, "base/ours 至少提供一个"

    n = len(class_ids)
    if class_names is None:
        class_names = [f"class {c}" for c in class_ids]

    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    fig = plt.figure(figsize=(figsize_per_cell[0]*cols, figsize_per_cell[1]*rows))
    axes = fig.subplots(rows, cols, squeeze=False)

    # 先估计一个统一的 x_range 方便对比（用分位数裁剪，避免极端值撑开）
    all_vals = []
    for cid in class_ids:
        if base is not None:
            score = class_pixel_score_from_queries(base[0], base[1], cid, mode=mode)
            p, n_ = extract_pos_neg_values(score, gt, cid, ignore_index, max_pos, max_neg, seed)
            if p.size: all_vals.append(p)
            if n_.size: all_vals.append(n_)
        if ours is not None:
            score = class_pixel_score_from_queries(ours[0], ours[1], cid, mode=mode)
            p, n_ = extract_pos_neg_values(score, gt, cid, ignore_index, max_pos, max_neg, seed+999)
            if p.size: all_vals.append(p)
            if n_.size: all_vals.append(n_)

    x_range = None
    if all_vals:
        concat = np.concatenate(all_vals, axis=0)
        lo = np.percentile(concat, 0.5)
        hi = np.percentile(concat, 99.5)
        x_range = (float(lo), float(hi))

    for i, (cid, cname) in enumerate(zip(class_ids, class_names)):
        r, c = divmod(i, cols)
        ax = axes[r][c]

        # baseline 虚线
        if base is not None:
            score_b = class_pixel_score_from_queries(base[0], base[1], cid, mode=mode)
            pos_b, neg_b = extract_pos_neg_values(score_b, gt, cid, ignore_index, max_pos, max_neg, seed)

            x1, y1 = _smooth_density_from_hist(pos_b, bins=bins, smooth_sigma=smooth_sigma, x_range=x_range)
            x2, y2 = _smooth_density_from_hist(neg_b, bins=bins, smooth_sigma=smooth_sigma, x_range=x_range)

            if x1 is not None: ax.plot(x1, y1, linestyle="--", linewidth=2.0, label="GT=class (base)")
            if x2 is not None: ax.plot(x2, y2, linestyle="--", linewidth=2.0, label="GT≠class (base)")

        # ours 实线
        if ours is not None:
            score_o = class_pixel_score_from_queries(ours[0], ours[1], cid, mode=mode)
            pos_o, neg_o = extract_pos_neg_values(score_o, gt, cid, ignore_index, max_pos, max_neg, seed+999)

            x1, y1 = _smooth_density_from_hist(pos_o, bins=bins, smooth_sigma=smooth_sigma, x_range=x_range)
            x2, y2 = _smooth_density_from_hist(neg_o, bins=bins, smooth_sigma=smooth_sigma, x_range=x_range)

            if x1 is not None: ax.plot(x1, y1, linestyle="-", linewidth=2.2, label="GT=class (ours)")
            if x2 is not None: ax.plot(x2, y2, linestyle="-", linewidth=2.2, label="GT≠class (ours)")

        ax.set_title(f"{cname} (id={cid})")
        ax.set_xlabel("score" if mode == "prob" else ("raw score" if mode == "raw" else "log(score)"))
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)
        if share_x and x_range is not None:
            ax.set_xlim(x_range)

    # 多余子图关掉
    for j in range(n, rows*cols):
        r, c = divmod(j, cols)
        axes[r][c].axis("off")

    if title is None:
        title = f"GT-conditioned Distribution ({mode})"
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        fig.savefig(save_path, dpi=200)
        print(f"[Saved] {save_path}")

    return fig






   