#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""MemoryBank-only ablation model.

目标：
1) 仅保留 memory bank（VideoQueryMemoryModule）的 init + forward 融合逻辑；
2) 不执行后续 ReferringTracker 的 track / indices 重排等操作；
3) loss 正常使用 self.criterion（utils/criterion_track.py::SetCriterion）；
4) forward 接口尽量与 MaskFormerModel_track_memorybank.py 保持一致，便于直接替换做消融。

实现说明：
- backbone + sem_seg_head 的计算沿用原实现（eval + no_grad，作为 frozen feature extractor）。
- 先得到每帧的 pred_embds/pred_masks/pred_logits。
- 对前 m 帧做 memory_encoder.init；对后 n 帧用 memory_encoder 输出 fused query tokens。
- 用 fused tokens 作为“最外层 query 表示”，通过 predictor 的 class_embed/mask_embed
  重新生成后 n 帧的 pred_logits/pred_masks，并用 criterion 计算损失。
"""

import torch
from torch import nn
from addict import Dict

import einops

from .memory.memory_encorder import VideoQueryMemoryModule
from .backbone.resnet import ResNet, resnet_spec
from .backbone.swin import D2SwinTransformer
from .pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from .transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder

from utils.criterion_track import SetCriterion
from utils.matcher import HungarianMatcher


class MaskFormerHead(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()
        self.pixel_decoder = self.pixel_decoder_init(cfg, input_shape)
        self.predictor = self.predictor_init(cfg)

    def pixel_decoder_init(self, cfg, input_shape):
        common_stride = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        transformer_dropout = cfg.MODEL.MASK_FORMER.DROPOUT
        transformer_nheads = cfg.MODEL.MASK_FORMER.NHEADS
        transformer_dim_feedforward = 1024
        transformer_enc_layers = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS
        conv_dim = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        mask_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        transformer_in_features = cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES

        pixel_decoder = MSDeformAttnPixelDecoder(
            input_shape,
            transformer_dropout,
            transformer_nheads,
            transformer_dim_feedforward,
            transformer_enc_layers,
            conv_dim,
            mask_dim,
            transformer_in_features,
            common_stride,
        )
        return pixel_decoder

    def predictor_init(self, cfg):
        in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        hidden_dim = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        num_queries = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        nheads = cfg.MODEL.MASK_FORMER.NHEADS
        dim_feedforward = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        pre_norm = cfg.MODEL.MASK_FORMER.PRE_NORM
        mask_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        enforce_input_project = False
        mask_classification = True

        predictor = MultiScaleMaskedTransformerDecoder(
            in_channels,
            num_classes,
            mask_classification,
            hidden_dim,
            num_queries,
            nheads,
            dim_feedforward,
            dec_layers,
            pre_norm,
            mask_dim,
            enforce_input_project,
        )
        return predictor

    def forward(self, features, mask=None):
        mask_features, _, multi_scale_features = self.pixel_decoder.forward_features(features)
        predictions = self.predictor(multi_scale_features, mask_features, mask)
        return predictions


class MaskFormerModel_memorybank_only(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = self.build_backbone(cfg)
        self.sem_seg_head = MaskFormerHead(cfg, self.backbone_feature_shape)
        self.num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        self.iter = 0
        self.cfg = cfg

        self.criterion = self.build_criterion(cfg)

        # memory-bank clip split: m frames for init, n frames for processing
        mem_cfg = getattr(cfg.MODEL, "MEMORY_BANK", None)
        if mem_cfg is None:
            self.num_mem_frames = 3
            self.num_track_frames = 3
        else:
            self.num_mem_frames = int(getattr(mem_cfg, "NUM_MEM_FRAMES", mem_cfg.get("NUM_MEM_FRAMES", 3)))
            self.num_track_frames = int(getattr(mem_cfg, "NUM_TRACK_FRAMES", mem_cfg.get("NUM_TRACK_FRAMES", 3)))

        # 注意：VideoQueryMemoryModule 内部默认 Q=10，需与 cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 对齐
        self.memory_encoder = VideoQueryMemoryModule(topk=None, mem_frames=self.num_mem_frames)

    # -------------------- build blocks --------------------
    def build_criterion(self, cfg):
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = 1.0

        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]
        criterion = SetCriterion(
            self.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            device=torch.device("cuda", cfg.local_rank),
        )
        return criterion

    def build_backbone(self, cfg):
        model_type = cfg.MODEL.BACKBONE.TYPE
        if "resnet" in model_type:
            channels = [64, 128, 256, 512]
            if cfg.MODEL.RESNETS.DEPTH > 34:
                channels = [item * 4 for item in channels]
            backbone = ResNet(resnet_spec[model_type][0], resnet_spec[model_type][1])
            self.backbone_feature_shape = {}
            for i, channel in enumerate(channels):
                self.backbone_feature_shape[f"res{i+2}"] = Dict({"channel": channel, "stride": 2 ** (i + 2)})
        elif model_type == "swin":
            swin_depth = {
                "tiny": [2, 2, 6, 2],
                "small": [2, 2, 18, 2],
                "base": [2, 2, 18, 2],
                "large": [2, 2, 18, 2],
            }
            swin_heads = {
                "tiny": [3, 6, 12, 24],
                "small": [3, 6, 12, 24],
                "base": [4, 8, 16, 32],
                "large": [6, 12, 24, 48],
            }
            swin_dim = {"tiny": 96, "small": 96, "base": 128, "large": 192}
            swin_window_size = {"tiny": 7, "small": 7, "base": 12, "large": 12}
            cfg.MODEL.SWIN.DEPTHS = swin_depth[cfg.MODEL.SWIN.TYPE]
            cfg.MODEL.SWIN.NUM_HEADS = swin_heads[cfg.MODEL.SWIN.TYPE]
            cfg.MODEL.SWIN.EMBED_DIM = swin_dim[cfg.MODEL.SWIN.TYPE]
            cfg.MODEL.SWIN.WINDOW_SIZE = swin_window_size[cfg.MODEL.SWIN.TYPE]
            backbone = D2SwinTransformer(cfg)
            self.backbone_feature_shape = backbone.output_shape()
        else:
            raise NotImplementedError("Do not support model type!")
        return backbone

    # -------------------- helpers --------------------
    def frame_decoder_loss_reshape(self, outputs, targets):
        """将 (B,T,...) 的视频输出整理成 criterion 所需的 (B*T,...) 格式，并把 targets 拆成逐帧 list。"""
        outputs["pred_masks"] = einops.rearrange(outputs["pred_masks"], "b q t h w -> (b t) q () h w")
        outputs["pred_logits"] = einops.rearrange(outputs["pred_logits"], "b t q c -> (b t) q c")

        if "aux_outputs" in outputs:
            for i in range(len(outputs["aux_outputs"])):
                outputs["aux_outputs"][i]["pred_masks"] = einops.rearrange(
                    outputs["aux_outputs"][i]["pred_masks"], "b q t h w -> (b t) q () h w"
                )
                outputs["aux_outputs"][i]["pred_logits"] = einops.rearrange(
                    outputs["aux_outputs"][i]["pred_logits"], "b t q c -> (b t) q c"
                )

        gt_instances = []
        for targets_per_video in targets:
            num_labeled_frames = targets_per_video["ids"].shape[1]
            for f in range(num_labeled_frames):
                labels = targets_per_video["labels"]
                ids = targets_per_video["ids"][:, [f]]
                masks = targets_per_video["masks"][:, [f], :, :]
                gt_instances.append({"labels": labels, "ids": ids, "masks": masks})

        return outputs, gt_instances

    def _get_binary_mask(self, target):
        y, x = target.size()
        target = target.to(torch.int64)
        target_onehot = torch.zeros(self.num_classes + 1, y, x, device=target.device)
        target_onehot = target_onehot.scatter(dim=0, index=target.unsqueeze(0), value=1)
        return target_onehot

    def _get_targets(self, gt_masks):
        """沿用 track_memorybank.py 的 targets 构造：仅保留在所有帧都出现的类。"""
        targets = []
        num_classes = 8

        for batch in gt_masks:  # (T,H,W)
            T, H, W = batch.shape
            counts = []
            flat = batch.view(T, -1)
            for t in range(T):
                cnt = torch.bincount(flat[t], minlength=num_classes)
                counts.append(cnt)
            counts = torch.stack(counts, dim=0)

            present = counts > 0
            present_all = present.all(dim=0)
            present_all[0] = False

            labels = torch.nonzero(present_all, as_tuple=False).squeeze(1)
            N = len(labels)
            ids = labels.repeat(T, 1).transpose(0, 1)

            gt_masks_per_video = torch.zeros((T, N, H, W), dtype=torch.bool, device=gt_masks.device)
            for t in range(T):
                binary_masks = self._get_binary_mask(batch[t])
                gt_masks_per_video[t] = binary_masks[labels]

            targets.append(
                {
                    "masks": einops.rearrange(gt_masks_per_video, "t n h w -> n t h w").float(),
                    "labels": labels,
                    "ids": ids,
                }
            )
        return targets

    # -------------------- forward --------------------
    def forward(self, inputs, targets_ori, istrain, name_idx, epoch, i, name):
        """inputs: (B,T,C,H,W)

        语义：
        - 前 m 帧：只用于 memory bank init；
        - 后 n 帧：用 memory bank 融合后的 query tokens 重新预测 mask/logits，并计算 loss。
        """

        self.backbone.eval()
        self.sem_seg_head.eval()

        B, T, _, _, _ = inputs.shape
        m = int(self.num_mem_frames)
        n = int(self.num_track_frames)
        if m <= 0 or n <= 0:
            raise ValueError(f"NUM_MEM_FRAMES(m) and NUM_TRACK_FRAMES(n) must be > 0, got m={m}, n={n}")
        if T < m + n:
            raise ValueError(f"Need at least T >= m+n frames, got T={T}, m={m}, n={n}")

        mem_slice = slice(0, m)
        track_slice = slice(m, m + n)
        T_track = n

        with torch.no_grad():
            inputs_flat = einops.rearrange(inputs, "b t c h w -> (b t) c h w")
            features = self.backbone(inputs_flat)
            image_outputs = self.sem_seg_head(features)

            # ---- rearrange to video layout ----
            image_outputs["mask_features"] = einops.rearrange(
                image_outputs["mask_features"], "(b t) c h w -> b t c h w", t=T
            )
            image_outputs["pred_logits"] = einops.rearrange(
                image_outputs["pred_logits"], "(b t) q c -> b t q c", t=T
            )
            image_outputs["pred_masks"] = einops.rearrange(
                image_outputs["pred_masks"], "(b t) q h w -> b q t h w", t=T
            )
            image_outputs["pred_embds"] = einops.rearrange(
                image_outputs["pred_embds"], "(b t) c q -> b c t q", t=T
            )

            if "aux_outputs" in image_outputs:
                for k in range(len(image_outputs["aux_outputs"])):
                    image_outputs["aux_outputs"][k]["pred_logits"] = einops.rearrange(
                        image_outputs["aux_outputs"][k]["pred_logits"], "(b t) q c -> b t q c", t=T
                    )
                    image_outputs["aux_outputs"][k]["pred_masks"] = einops.rearrange(
                        image_outputs["aux_outputs"][k]["pred_masks"], "(b t) q h w -> b q t h w", t=T
                    )

            # memory inputs
            frame_embds_all = image_outputs["pred_embds"].detach()  # (B,C,T,Q)
            mask_all = image_outputs["pred_masks"].detach()         # (B,Q,T,H,W)
            logits_all = image_outputs["pred_logits"].detach()      # (B,T,Q,C+1)

            frame_embds_track = frame_embds_all[:, :, track_slice, :]  # (B,C,n,Q)
            mask_features_track = image_outputs["mask_features"][:, track_slice, :, :, :]  # (B,n,Cm,Hm,Wm)

            # aux sliced (optional, keep for deep supervision)
            aux_outputs_track = None
            if "aux_outputs" in image_outputs:
                aux_outputs_track = []
                for k in range(len(image_outputs["aux_outputs"])):
                    aux = image_outputs["aux_outputs"][k]
                    aux_outputs_track.append(
                        {
                            "pred_logits": aux["pred_logits"][:, track_slice, :, :],
                            "pred_masks": aux["pred_masks"][:, :, track_slice, :, :],
                        }
                    )

        # 释放大字典（降低显存峰值）
        try:
            del image_outputs
        except Exception:
            pass

        # ---- memory bank only ----
        init_frame_ids = list(range(0, m))
        proc_frame_ids = list(range(m, m + n))

        self.memory_encoder.init(frame_embds_all, mask_all, logits_all, init_frame_ids=init_frame_ids)
        fused_embds = self.memory_encoder(
            frame_embds_all,
            mask_all,
            frame_embds_track,
            logits_all,
            proc_frame_ids=proc_frame_ids,
        )  # (B,C,n,Q)

        # ---- reconstruct predictions for track frames from fused tokens ----
        # fused: (B,C,n,Q) -> (B,n,Q,C)
        fused_bnqc = fused_embds.permute(0, 2, 3, 1).contiguous()

        predictor = self.sem_seg_head.predictor
        pred_logits_track = predictor.class_embed(fused_bnqc)  # (B,n,Q,C+1)

        mask_embed_track = predictor.mask_embed(fused_bnqc)  # (B,n,Q,mask_dim)
        # mask_features_track: (B,n,mask_dim,H,W)
        pred_masks_bnqhw = torch.einsum("bnqd,bndhw->bnqhw", mask_embed_track, mask_features_track)
        pred_masks_bqnhw = pred_masks_bnqhw.permute(0, 2, 1, 3, 4).contiguous()  # (B,Q,n,H,W)

        outputs_track = {
            "pred_logits": pred_logits_track,
            "pred_masks": pred_masks_bqnhw,
        }
        if aux_outputs_track is not None:
            outputs_track["aux_outputs"] = aux_outputs_track

        # ---- targets & loss ----
        targets_track_ori = targets_ori[:, track_slice, :, :]
        targets = self._get_targets(targets_track_ori)

        outputs_for_loss, gt_instances = self.frame_decoder_loss_reshape(outputs_track, targets)
        # 直接用 criterion（内部含 matcher）
        losses, _ = self.criterion(outputs_for_loss, targets_track_ori, matcher_outputs=None)

        # ---- reshape outputs for training loop (b,t,q,...) ----
        outputs_reshape = {
            "pred_logits": einops.rearrange(outputs_for_loss["pred_logits"], "(b t) q c -> b t q c", t=T_track),
            "pred_masks": einops.rearrange(outputs_for_loss["pred_masks"], "(b t) q () h w -> b t q h w", t=T_track),
        }

        if istrain:
            return outputs_reshape, losses
        else:
            return outputs_reshape, losses
