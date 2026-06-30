

import torch
from torch import nn
from addict import Dict
from .memory.memory_encorder import VideoQueryMemoryModule
from .backbone.swin import D2SwinTransformer
from .pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from .transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder

from .dvis.video_dvis_modules_ori import ReferringTracker, TemporalRefiner
from utils.criterion_track import SetCriterion, Criterion
from utils.matcher import HungarianMatcher
import einops
import os
import json
from collections import deque

def ensure_list(input_data):

    if isinstance(input_data, list):
        return input_data
    elif isinstance(input_data, str):
        return [input_data]
    else:
        raise TypeError("Input must be a string or a list")

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
        "names": ensure_list(name_list),
        "indice_dict": indice_dict_to_serializable(indice_dict),
    }
    if extra is not None:
        record["extra"] = extra

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()

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
        transformer_in_features =  cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES

        pixel_decoder = MSDeformAttnPixelDecoder(input_shape,
                                                transformer_dropout,
                                                transformer_nheads,
                                                transformer_dim_feedforward,
                                                transformer_enc_layers,
                                                conv_dim,
                                                mask_dim,
                                                transformer_in_features,
                                                common_stride)
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
        predictor = MultiScaleMaskedTransformerDecoder(in_channels,
                                                        num_classes,
                                                        mask_classification,
                                                        hidden_dim,
                                                        num_queries,
                                                        nheads,
                                                        dim_feedforward,
                                                        dec_layers,
                                                        pre_norm,
                                                        mask_dim,
                                                        enforce_input_project)
        return predictor

    def forward(self, features, mask=None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)
        predictions = self.predictor(multi_scale_features, mask_features,mask)
        return predictions

class MaskFormerModel_track_memorybank(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = self.build_backbone(cfg)
        self.sem_seg_head = MaskFormerHead(cfg, self.backbone_feature_shape)
        self.num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        self.iter=0
        self.cfg=cfg
        self.tracker = self.build_tracker(cfg)

        if bool(getattr(cfg, "efficiency", False)) or bool(getattr(cfg, "inferonly", False)):
            self.criterion = None
        else:
            self.criterion = self.build_criterion(cfg)

        mem_cfg = getattr(cfg.MODEL, "MEMORY_BANK", None)
        if mem_cfg is None:
            self.num_mem_frames = 3
            self.num_track_frames = 3
        else:
            self.num_mem_frames = int(getattr(mem_cfg, "NUM_MEM_FRAMES", mem_cfg.get("NUM_MEM_FRAMES", 3)))
            self.num_track_frames = int(getattr(mem_cfg, "NUM_TRACK_FRAMES", mem_cfg.get("NUM_TRACK_FRAMES", 3)))
        self.memory_encoder = VideoQueryMemoryModule(topk=None, mem_frames=self.num_mem_frames)

    @torch.no_grad()
    def forward_infer(
        self,
        inputs: torch.Tensor,
        use_memory_bank: bool = True,
        return_intermediates: bool = False,
    ):

        self.keep = False
        self.backbone.eval()
        self.sem_seg_head.eval()
        self.tracker.eval()

        B, T, C_in, H, W = inputs.shape
        m = int(self.num_mem_frames)
        n = int(self.num_track_frames)
        if m <= 0 or n <= 0:
            raise ValueError(f"NUM_MEM_FRAMES(m) and NUM_TRACK_FRAMES(n) must be > 0, got m={m}, n={n}")
        if T < m + n:
            raise ValueError(f"Need at least T >= m+n frames, got T={T}, m={m}, n={n}")

        # Context frames populate memory; later frames receive tracking supervision.
        mem_slice = slice(0, m)
        track_slice = slice(m, m + n)
        T_track = n

        inputs_flat = einops.rearrange(inputs, 'b t c h w -> (b t) c h w')
        features = self.backbone(inputs_flat)
        image_outputs = self.sem_seg_head(features)

        image_outputs['mask_features'] = einops.rearrange(
            image_outputs['mask_features'], '(b t) c h w -> b t c h w', t=T
        )
        image_outputs['pred_logits'] = einops.rearrange(
            image_outputs['pred_logits'], '(b t) q c -> b t q c', t=T
        )
        image_outputs['pred_masks'] = einops.rearrange(
            image_outputs['pred_masks'], '(b t) q h w -> b q t h w', t=T
        )
        image_outputs['pred_embds'] = einops.rearrange(
            image_outputs['pred_embds'], '(b t) c q -> b c t q', t=T
        )

        frame_embds_all = image_outputs['pred_embds'].detach()
        mask_all = image_outputs['pred_masks'].detach()
        logits_all = image_outputs['pred_logits'].detach()

        frame_embds_track = frame_embds_all[:, :, track_slice, :]
        mask_feature_track = image_outputs['mask_features'][:, track_slice, :, :, :]

        if use_memory_bank:
            init_frame_ids = list(range(0, m))
            proc_frame_ids = list(range(m, m + n))
            self.memory_encoder.init(
                frame_embds_all, mask_all, logits_all, init_frame_ids=init_frame_ids
            )
            frame_embds = self.memory_encoder(
                frame_embds_all,
                mask_all,
                frame_embds_track,
                logits_all,
                proc_frame_ids=proc_frame_ids,
            )
        else:
            frame_embds = frame_embds_track

        outputs_ori = self.tracker(
            frame_embds,
            mask_feature_track,
            return_indices=False,
            resume=self.keep,
        )

        out = {}
        if isinstance(outputs_ori, dict) and ('pred_logits' in outputs_ori) and ('pred_masks' in outputs_ori):
            pl = outputs_ori['pred_logits']
            pm = outputs_ori['pred_masks']

            if pl.dim() == 4:

                out['pred_logits'] = pl
            elif pl.dim() == 3:
                out['pred_logits'] = einops.rearrange(pl, '(b t) q c -> b t q c', t=T_track)
            else:
                out['pred_logits'] = pl

            if pm.dim() == 5:

                out['pred_masks'] = pm
            elif pm.dim() == 6:

                if pm.shape[2] == 1:
                    out['pred_masks'] = einops.rearrange(pm, '(b t) q () h w -> b t q h w', t=T_track)
                else:

                    out['pred_masks'] = pm.squeeze(3)
            else:
                out['pred_masks'] = pm

            for k, v in outputs_ori.items():
                if k in out:
                    continue
                out[k] = v
        else:
            out = outputs_ori

        if return_intermediates:
            return out, {
                "frame_embds_all": frame_embds_all,
                "mask_all": mask_all,
                "logits_all": logits_all,
            }
        return out

    def build_criterion(self, cfg):

        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION

        no_object_weight = 1.0

        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        boundary_weight = cfg.MODEL.MASK_FORMER.BOUNDARY_WEIGHT

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
            device=torch.device("cuda", cfg.local_rank)
        )

        return criterion

    def build_tracker(self, cfg):

        tracker = ReferringTracker(
            hidden_channel=cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            feedforward_channel=cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD,
            num_head=cfg.MODEL.MASK_FORMER.NHEADS,
            decoder_layer_num=cfg.MODEL.TRACKER.DECODER_LAYERS,
            mask_dim=cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            class_num=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
        )
        return tracker

    def build_backbone(self, cfg):
        model_type = cfg.MODEL.BACKBONE.TYPE
        if model_type == 'swin':
            swin_depth = {'tiny': [2, 2, 6, 2], 'small': [2, 2, 18, 2], 'base': [2, 2, 18, 2], 'large': [2, 2, 18, 2]}
            swin_heads = {'tiny': [3, 6, 12, 24], 'small': [3, 6, 12, 24], 'base': [4, 8, 16, 32], 'large': [6, 12, 24, 48]}
            swin_dim = {'tiny':96, 'small': 96, 'base': 128, 'large': 192}
            swin_window_size = {'tiny': 7, 'small': 7, 'base': 12, 'large': 12}
            cfg.MODEL.SWIN.DEPTHS = swin_depth[cfg.MODEL.SWIN.TYPE]
            cfg.MODEL.SWIN.NUM_HEADS = swin_heads[cfg.MODEL.SWIN.TYPE]
            cfg.MODEL.SWIN.EMBED_DIM = swin_dim[cfg.MODEL.SWIN.TYPE]
            cfg.MODEL.SWIN.WINDOW_SIZE = swin_window_size[cfg.MODEL.SWIN.TYPE]
            backbone = D2SwinTransformer(cfg)
            self.backbone_feature_shape = backbone.output_shape()
        else:
            raise NotImplementedError(f"Unsupported backbone: {model_type}. TEDi uses Swin Transformer.")
        return backbone

    def forward(self, inputs, targets_ori, istrain, name_idx, epoch, i, name):

        self.keep = False
        self.backbone.eval()
        self.sem_seg_head.eval()

        B, T, C_in, H, W = inputs.shape
        m = int(self.num_mem_frames)
        n = int(self.num_track_frames)
        if m <= 0 or n <= 0:
            raise ValueError(f"NUM_MEM_FRAMES(m) and NUM_TRACK_FRAMES(n) must be > 0, got m={m}, n={n}")
        if T < m + n:
            raise ValueError(f"Need at least T >= m+n frames, got T={T}, m={m}, n={n}")

        # Context frames populate memory; later frames receive tracking supervision.
        mem_slice = slice(0, m)
        track_slice = slice(m, m + n)
        T_track = n
        output_track={}
        with torch.no_grad():

            inputs_flat = einops.rearrange(inputs, 'b t c h w -> (b t) c h w')
            features = self.backbone(inputs_flat)
            image_outputs = self.sem_seg_head(features)

            image_outputs['mask_features'] = einops.rearrange(
                image_outputs['mask_features'], '(b t) c h w -> b t c h w', t=T
            )
            image_outputs['pred_logits'] = einops.rearrange(
                image_outputs['pred_logits'], '(b t) q c -> b t q c', t=T
            )
            image_outputs['pred_masks'] = einops.rearrange(
                image_outputs['pred_masks'], '(b t) q h w -> b q t h w', t=T
            )
            image_outputs['pred_embds'] = einops.rearrange(
                image_outputs['pred_embds'], '(b t) c q -> b c t q', t=T
            )

            frame_embds_all = image_outputs['pred_embds'].detach()
            mask_all = image_outputs['pred_masks'].detach()
            logits_all = image_outputs['pred_logits'].detach()
            frame_embds_track= frame_embds_all[:, :, track_slice, :]
            mask_feature_track = image_outputs['mask_features'][:, track_slice, :, :, :]

            if 'aux_outputs' in image_outputs:
                for k in range(len(image_outputs['aux_outputs'])):
                    image_outputs['aux_outputs'][k]['pred_logits'] = einops.rearrange(
                        image_outputs['aux_outputs'][k]['pred_logits'], '(b t) q c -> b t q c', t=T
                    )
                    image_outputs['aux_outputs'][k]['pred_masks'] = einops.rearrange(
                        image_outputs['aux_outputs'][k]['pred_masks'], '(b t) q h w -> b q t h w', t=T
                    )
            output_track['mask_features']=image_outputs['mask_features'][:, track_slice, :, :, :]
            output_track['pred_logits']=image_outputs['pred_logits'][:, track_slice, :, :]
            output_track['pred_masks']=image_outputs['pred_masks'][:, :, track_slice, :, :]
            output_track['pred_embds']=image_outputs['pred_embds'][:, :, track_slice, :]
            if 'aux_outputs' in image_outputs:

                output_track['aux_outputs'] = []

                for k in range(len(image_outputs['aux_outputs'])):

                    aux_output = image_outputs['aux_outputs'][k]

                    output_track['aux_outputs'].append({
                        'pred_logits': aux_output['pred_logits'][:, track_slice, :, :],
                        'pred_masks': aux_output['pred_masks'][:, :, track_slice, :, :]
                    })
            del image_outputs

        init_frame_ids = list(range(0, m))
        proc_frame_ids = list(range(m, m + n))

        self.memory_encoder.init(frame_embds_all, mask_all, logits_all, init_frame_ids=init_frame_ids)

        frame_embds = self.memory_encoder(
            frame_embds_all,
            mask_all,
            frame_embds_track,
            logits_all,
            proc_frame_ids=proc_frame_ids,
        )

        outputs_ori, indices = self.tracker(
            frame_embds,
            mask_feature_track,
            return_indices=True,
            resume=self.keep,
        )

        output_track = self.reset_image_output_order(output_track, indices)
        targets_track_ori=targets_ori[:, track_slice, :, :]

        targets = self._get_targets(targets_track_ori)

        if self.iter < self.cfg.SOLVER.MAX_ITER // 2:
            output_track, outputs, targets = self.frame_decoder_loss_reshape(
                outputs_ori, targets, output_track=output_track
            )
        else:
            output_track, outputs, targets = self.frame_decoder_loss_reshape(
                outputs_ori, targets, output_track=None
            )

        idx_list = name_idx.detach().cpu().tolist()
        cur_names = [name[j] for j in idx_list]

        if self.criterion is None:
            self.criterion = self.build_criterion(self.cfg)
        losses, indice_dict = self.criterion(
            outputs,
            targets_track_ori.squeeze(0),
            matcher_outputs=output_track,
        )

        outputs_reshape = {}
        outputs_reshape['pred_logits'] = einops.rearrange(
            outputs['pred_logits'], ' (b t) q c -> b t q c', t=T_track
        )
        outputs_reshape['pred_masks'] = einops.rearrange(
            outputs['pred_masks'], ' (b t) q () h w -> b t q h w', t=T_track
        )

        if istrain:
            return outputs_reshape, losses
        else:
            outputs_ori['pred_logits'] = einops.rearrange(
                outputs_ori['pred_logits'], ' (b t) q c -> b t q c', t=T_track
            )
            outputs_ori['pred_masks'] = einops.rearrange(
                outputs_ori['pred_masks'], ' (b t) q () h w -> b t q h w', t=T_track
            )

            return outputs_ori, losses

    def _get_targets(self, gt_masks):
        targets = []
        num_classes = 8

        for batch in gt_masks:
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

            mask_shape = [T, N, H, W]
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool,device=gt_masks.device)

            for t in range(T):
                binary_masks = self._get_binary_mask(batch[t])
                gt_masks_per_video[t] = binary_masks[labels]

            targets.append({'masks': einops.rearrange(gt_masks_per_video, 't n h w -> n t h w').float(), 'labels': labels, 'ids': ids})

        return targets

    def _get_binary_mask(self, target):

        y, x = target.size()
        target=target.to(torch.int64)
        target_onehot = torch.zeros(self.num_classes + 1, y, x).to(device=target.device, non_blocking=True)
        target_onehot = target_onehot.scatter(dim=0, index=target.unsqueeze(0), value=1)
        return target_onehot

    def frame_decoder_loss_reshape(self, outputs, targets, output_track=None):
        outputs['pred_masks'] = einops.rearrange(outputs['pred_masks'], 'b q t h w -> (b t) q () h w')
        outputs['pred_logits'] = einops.rearrange(outputs['pred_logits'], 'b t q c -> (b t) q c')
        if output_track is not None:
            output_track['pred_masks'] = einops.rearrange(output_track['pred_masks'], 'b q t h w -> (b t) q () h w')
            output_track['pred_logits'] = einops.rearrange(output_track['pred_logits'], 'b t q c -> (b t) q c')
        if 'aux_outputs' in outputs:
            for i in range(len(outputs['aux_outputs'])):
                outputs['aux_outputs'][i]['pred_masks'] = einops.rearrange(
                    outputs['aux_outputs'][i]['pred_masks'], 'b q t h w -> (b t) q () h w'
                )
                outputs['aux_outputs'][i]['pred_logits'] = einops.rearrange(
                    outputs['aux_outputs'][i]['pred_logits'], 'b t q c -> (b t) q c'
                )
        gt_instances = []
        for targets_per_video in targets:
            num_labeled_frames = targets_per_video['ids'].shape[1]
            for f in range(num_labeled_frames):
                labels = targets_per_video['labels']
                ids = targets_per_video['ids'][:, [f]]
                masks = targets_per_video['masks'][:, [f], :, :]

                gt_instances.append({"labels": labels, "ids": ids, "masks": masks})
        return output_track, outputs, gt_instances

    def convert_indices_list(self,indices_list):

        idx_tbq = torch.stack(
            [torch.as_tensor(x, dtype=torch.long) for x in indices_list],
            dim=0
        )

        idx_btq = idx_tbq.permute(1, 0, 2).contiguous()

        new_indices_list = [idx_btq[b] for b in range(idx_btq.size(0))]

        return new_indices_list

    def reset_image_output_order(self, output, indices_list):

        indices_list=self.convert_indices_list(indices_list)
        for i in range(len(indices_list)):
            indices = torch.Tensor(indices_list[i]).to(torch.int64)
            frame_indices = torch.range(0, indices.shape[0] - 1).to(indices).unsqueeze(1).repeat(1, indices.shape[1])

            output['pred_masks'][i] = output['pred_masks'][i][indices, frame_indices].transpose(0, 1)

            output['pred_logits'][i] = output['pred_logits'][i][frame_indices, indices]
        return output

    def post_processing(self, outputs, aux_logits=None):

        pred_logits = outputs['pred_logits']
        pred_logits = pred_logits[0]
        out_logits = torch.mean(pred_logits, dim=0).unsqueeze(0)
        if aux_logits is not None:
            aux_logits = aux_logits[0]
            aux_logits = torch.mean(aux_logits, dim=0)
        outputs['pred_logits'] = out_logits
        outputs['ids'] = [torch.arange(0, outputs['pred_masks'].size(1))]
        if aux_logits is not None:
            return outputs, aux_logits
        return outputs
