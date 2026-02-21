#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   MaskFormerModel.py
@Time    :   2022/09/30 20:50:53
@Author  :   BQH 
@Version :   1.0
@Contact :   raogx.vip@hotmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   基于DeformTransAtten的分割网络
'''

# here put the import lib
import torch
from torch import nn
from addict import Dict

from .backbone.resnet import ResNet, resnet_spec
from .backbone.swin import D2SwinTransformer
from .pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from .transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder
from .dvis.meta_architecture import DVIS_online,DVIS_offline
from .dvis.video_dvis_modules_ori import ReferringTracker, TemporalRefiner
import sys
sys.path.append('/home/yjh/code_yjh_bishe/DVIS-main/')
from mask2former_video.modeling.criterion import VideoSetCriterion
from mask2former_video.modeling.matcher import VideoHungarianMatcher, VideoHungarianMatcher_Consistent
from utils.criterion_track import SetCriterion, Criterion
from utils.matcher import HungarianMatcher
import einops
import os
import json
#保存indice
def ensure_list(input_data):
    # 判断输入是否是一个 list
    if isinstance(input_data, list):
        return input_data  # 如果是 list，直接返回
    elif isinstance(input_data, str):
        return [input_data]  # 如果是 string，将其放入 list 中返回
    else:
        raise TypeError("Input must be a string or a list")  # 如果是其他类型，抛出错误

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
        "names": ensure_list(name_list),  # 你的 name list
        "indice_dict": indice_dict_to_serializable(indice_dict),
    }
    if extra is not None:
        record["extra"] = extra  # 你想加别的信息：loss, lr, etc.

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
        transformer_in_features =  cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES # ["res3", "res4", "res5"]

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
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)     #先经过 pixel_decoder再经过transformer_decoder
        predictions = self.predictor(multi_scale_features, mask_features,mask)        
        return predictions

class MaskFormerModel_track(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = self.build_backbone(cfg)
        self.sem_seg_head = MaskFormerHead(cfg, self.backbone_feature_shape)
        self.num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        self.iter=0
        self.cfg=cfg
        self.tracker = self.build_tracker(cfg)
        self.criterion = self.build_criterion(cfg)
        
    def build_criterion(self, cfg):
         # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        # no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT
        no_object_weight = 1.0
        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        boundary_weight = cfg.MODEL.MASK_FORMER.BOUNDARY_WEIGHT

            # building criterion
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
        # building criterion
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
        if  'resnet' in model_type:            
            channels = [64, 128, 256, 512]
            if cfg.MODEL.RESNETS.DEPTH > 34:
                channels = [item * 4 for item in channels] # [256, 512, 1024, 2048]
            backbone = ResNet(resnet_spec[model_type][0], resnet_spec[model_type][1])
            # backbone.init_weights()
            self.backbone_feature_shape = dict()
            for i, channel in enumerate(channels):
                self.backbone_feature_shape[f'res{i+2}'] = Dict({'channel': channel, 'stride': 2**(i+2)})
        elif model_type == 'swin':
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
            raise NotImplementedError('Do not support model type!')
        return backbone

    def forward(self, inputs,targets_ori,istrain,name_idx,epoch, i, name):
        self.keep = False
        self.backbone.eval()
        self.sem_seg_head.eval()                                     
        with torch.no_grad():
            inputs=einops.rearrange(inputs, 'b t c h w -> (b t) c h w')
            features = self.backbone(inputs)
            image_outputs = self.sem_seg_head(features)
            image_outputs['mask_features'] =einops.rearrange(image_outputs['mask_features'], '(b t) c h w -> b t c h w', t=3)
            image_outputs['pred_logits'] =einops.rearrange(image_outputs['pred_logits'], '(b t) q c -> b t q c', t=3)
            image_outputs['pred_masks'] = einops.rearrange(image_outputs['pred_masks'], '(b t) q h w -> b q t h w', t=3)
            image_outputs['pred_embds'] = einops.rearrange(image_outputs['pred_embds'], '(b t) c q -> b c t q', t=3)
            
            # frame_embds = image_outputs['pred_embds'].clone().detach()  # (b, c, t,q)
            # mask_features = image_outputs['mask_features'].clone().detach()
            # pred_masks = image_outputs['pred_masks'].clone().detach()
              # 提取所需的部分
            
            frame_embds = image_outputs['pred_embds'].detach()  # (b, c, t, q)
            mask_features = image_outputs['mask_features'].detach()
            
    # 结果为 (b, q, t)，每个 (b, q, t) 表示该位置在左侧还是右侧
            
    # def forward(self, mask_features, pred_embds, pred_logits, pred_masks,targets_ori,istrain):
    #     self.keep = False
    #     image_outputs = {}
    #     image_outputs['mask_features'] =mask_features
    #     image_outputs['pred_logits'] =pred_logits
    #     image_outputs['pred_masks'] = einops.rearrange(pred_masks, 'b t q h w -> b q t h w')
    #     image_outputs['pred_embds'] = einops.rearrange(pred_embds, 'b t c q -> b c t q')
    #     frame_embds = image_outputs['pred_embds'].clone().detach()  # (b, c, t,q)
    #     mask_features = image_outputs['mask_features'].clone().detach()
        
        
        # features = self.backbone(inputs[:,0,...]) #输出不同尺寸的特征图（res2，res3，res4，res5） 
        # image_outputs = self.sem_seg_head(features)
        # frame_embds = image_outputs['pred_embds'].clone().detach()  # (b, c, q)
        # mask_features = image_outputs['mask_features'].clone().detach().unsqueeze(0)
        # pred_logits= image_outputs['pred_logits'].clone()
        # pred_masks = image_outputs['pred_masks'].clone()
        # features_previous = self.backbone(inputs[:,1,...]) #输出不同尺寸的特征图（res2，res3，res4，res5）
        # image_outputs_previous = self.sem_seg_head(features_previous)
        # frame_embds_previous = image_outputs_previous['pred_embds'].clone().detach()  # (b, c, q)
        # mask_features_previous = image_outputs_previous['mask_features'].clone().detach().unsqueeze(0)
        # pred_logits_previous= image_outputs_previous['pred_logits'].clone()
        # pred_masks_previous = image_outputs_previous['pred_masks'].clone()
        # image_outputs['pred_logits']=torch.stack([pred_logits,pred_logits_previous],dim=1) #(b, t, q, c)
        # image_outputs['pred_masks']=torch.stack([pred_masks,pred_masks_previous],dim=2) #(b, q, t, h, w)
        
        # if 'aux_outputs' in image_outputs:
        #     for i in range(len(image_outputs['aux_outputs'])):
        #         image_outputs['aux_outputs'][i]['pred_logits']=torch.stack([image_outputs['aux_outputs'][i]['pred_logits'],image_outputs['aux_outputs'][i]['pred_logits']],dim=1)
        #         image_outputs['aux_outputs'][i]['pred_masks']=torch.stack([image_outputs['aux_outputs'][i]['pred_masks'],image_outputs['aux_outputs'][i]['pred_masks']],dim=1)
        
        if 'aux_outputs' in image_outputs:
            for i in range(len(image_outputs['aux_outputs'])):
                image_outputs['aux_outputs'][i]['pred_logits']= einops.rearrange(image_outputs['aux_outputs'][i]['pred_logits'], '(b t) q c -> b t q c', t=3)
                image_outputs['aux_outputs'][i]['pred_masks']= einops.rearrange(image_outputs['aux_outputs'][i]['pred_masks'], '(b t) q h w -> b q t h w', t=3)
        del image_outputs['mask_features']
        # torch.cuda.empty_cache()
        
        outputs_ori, indices = self.tracker(frame_embds, mask_features, return_indices=True, resume=self.keep) #outoutput是tracker的输出
        image_outputs = self.reset_image_output_order(image_outputs, indices)  # image_output是semseg的输出并经过重新排列
        
        targets = self._get_targets(targets_ori)
        # print(targets_ori.max())
        # use the segmenter prediction results to guide the matching process during early training phase
        if self.iter < self.cfg.SOLVER.MAX_ITER // 2:
            image_outputs, outputs, targets = self.frame_decoder_loss_reshape(
                outputs_ori, targets, image_outputs=image_outputs
            )
        
        else:
            image_outputs, outputs, targets = self.frame_decoder_loss_reshape(
                outputs_ori, targets, image_outputs=None
            )
        # print(self.iter)
        idx_list = name_idx.detach().cpu().tolist()   # 例如 [12, 13, 99, 3]
        cur_names = [name[j] for j in idx_list]       # 取出 batch 的名字列表
        losses,indice_dict = self.criterion(outputs, targets_ori.squeeze(0), matcher_outputs=image_outputs) #此处在最开始用image_outputs用于匹配，一半的iter后用outputs用作匹配
        if istrain:
            append_epoch_jsonl("/home/yjh/code_yjh_bishe/Mask2Former-Simplify-master/yjh/indice/train/{}".format(self.cfg.task), epoch, i, cur_names, indice_dict)
        else:
            append_epoch_jsonl("/home/yjh/code_yjh_bishe/Mask2Former-Simplify-master/yjh/indice/test/{}".format(self.cfg.task), epoch, i, cur_names, indice_dict)
        outputs_reshape={}
        outputs_reshape['pred_logits']=einops.rearrange(outputs['pred_logits'], ' (b t) q c -> b t q c ',t=3)
        outputs_reshape['pred_masks']=einops.rearrange(outputs['pred_masks'], ' (b t) q () h w -> b t q h w ',t=3)
        

        if istrain:
            return outputs_reshape,losses
        else:
            outputs_ori['pred_logits']=einops.rearrange(outputs_ori['pred_logits'], ' (b t) q c -> b t q c ',t=3)
            outputs_ori['pred_masks']=einops.rearrange(outputs_ori['pred_masks'], ' (b t) q () h w -> b t q h w ',t=3)
            return outputs_ori,losses
       

        
        
        # losses = self.criterion(outputs, targets_ori.squeeze(0), matcher_outputs=image_outputs) #此处在最开始用image_outputs用于匹配，一半的iter后用outputs用作匹配
        # outputs_reshape={}
        # outputs_reshape['pred_logits']=einops.rearrange(outputs['pred_logits'], ' (b t) q c -> b t q c ',t=3)
        # outputs_reshape['pred_masks']=einops.rearrange(outputs['pred_masks'], ' (b t) q () h w -> b t q h w ',t=3)
        # if istrain:
        #     return outputs_reshape,losses
        # else:
        #     outputs_ori['pred_logits']=einops.rearrange(outputs_ori['pred_logits'], ' (b t) q c -> b t q c ',t=3)
        #     outputs_ori['pred_masks']=einops.rearrange(outputs_ori['pred_masks'], ' (b t) q () h w -> b t q h w ',t=3)
        #     return outputs_ori,losses
    def _get_targets(self, gt_masks):  # 输入: (B, T, H, W)
        targets = []
        num_classes = 8

        for batch in gt_masks:  # batch: (T, H, W)
            T, H, W = batch.shape

            # [T, C] 每帧统计每类像素数
            counts = []
            flat = batch.view(T, -1)
            for t in range(T):
                cnt = torch.bincount(flat[t], minlength=num_classes)
                counts.append(cnt)
            counts = torch.stack(counts, dim=0)  # [T, C]

            present = counts > 0
            present_all = present.all(dim=0)
            present_all[0] = False  # 背景不要

            labels = torch.nonzero(present_all, as_tuple=False).squeeze(1)  # [N]
            # print(labels)
            N = len(labels)
            ids = labels.repeat(T, 1).transpose(0, 1)  # (N, T)

            # 关键修正 ↓↓↓
            mask_shape = [T, N, H, W]
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool,device=gt_masks.device)

            for t in range(T):
                binary_masks = self._get_binary_mask(batch[t])  # (C, H, W)
                gt_masks_per_video[t] = binary_masks[labels]     # (N, H, W)
            #修改，batch=1时才成立

            # targets.append({'masks': einops.rearrange(gt_masks_per_video, 't n h w -> (n t) h w').float(), 'labels': labels, 'ids': ids})
            targets.append({'masks': einops.rearrange(gt_masks_per_video, 't n h w -> n t h w').float(), 'labels': labels, 'ids': ids})
            

        return targets


    def _get_binary_mask(self, target):
        # print(target.max(), target.min())
        y, x = target.size()
        target=target.to(torch.int64)
        target_onehot = torch.zeros(self.num_classes + 1, y, x).to(device=target.device, non_blocking=True)
        target_onehot = target_onehot.scatter(dim=0, index=target.unsqueeze(0), value=1)
        return target_onehot

    def frame_decoder_loss_reshape(self, outputs, targets, image_outputs=None):
        outputs['pred_masks'] = einops.rearrange(outputs['pred_masks'], 'b q t h w -> (b t) q () h w')
        outputs['pred_logits'] = einops.rearrange(outputs['pred_logits'], 'b t q c -> (b t) q c')
        if image_outputs is not None:
            image_outputs['pred_masks'] = einops.rearrange(image_outputs['pred_masks'], 'b q t h w -> (b t) q () h w')
            image_outputs['pred_logits'] = einops.rearrange(image_outputs['pred_logits'], 'b t q c -> (b t) q c')
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
        return image_outputs, outputs, gt_instances

    def convert_indices_list(self,indices_list):
        """
        indices_list: list length = t, each element shape (b, q)
        return:
            new_indices_list: list length = b, each element shape (t, q)
        """
        # 1) stack 成 (t, b, q)
        idx_tbq = torch.stack(
            [torch.as_tensor(x, dtype=torch.long) for x in indices_list],
            dim=0
        )  # (t, b, q)

        # 2) permute 成 (b, t, q)
        idx_btq = idx_tbq.permute(1, 0, 2).contiguous()  # (b, t, q)

        # 3) 拆成 list[b]，每个是 (t, q)
        new_indices_list = [idx_btq[b] for b in range(idx_btq.size(0))]

        return new_indices_list

    def reset_image_output_order(self, output, indices_list):
        """
        in order to maintain consistency between the initial query and the guided results (segmenter prediction)
        :param output: segmenter prediction results (image-level segmentation results)
        :param indices: matched indicates
        :return: reordered outputs
        """
        indices_list=self.convert_indices_list(indices_list)
        for i in range(len(indices_list)):
            indices = torch.Tensor(indices_list[i]).to(torch.int64)  # (t, q)
            frame_indices = torch.range(0, indices.shape[0] - 1).to(indices).unsqueeze(1).repeat(1, indices.shape[1])
            # print(indices.shape, frame_indices.shape)
            # print(output['pred_masks'].shape)
            # # pred_masks, shape is (b, q, t, h, w)
            # print(output['pred_masks'][0].shape)
            output['pred_masks'][i] = output['pred_masks'][i][indices, frame_indices].transpose(0, 1) #[indices, frame_indices]表示的实际是原始的输出结果在[3,64]维度上每一个值重排后对应的索引
            # a=output['pred_masks'][0][indices, frame_indices]
            # print(output['pred_masks'][0].shape)
            # pred logits, shape is (b, t, q, c)
            output['pred_logits'][i] = output['pred_logits'][i][frame_indices, indices]
        return output

    def post_processing(self, outputs, aux_logits=None):
        """
        average the class logits and append query ids
        """
        pred_logits = outputs['pred_logits']
        pred_logits = pred_logits[0]  # (t, q, c)
        out_logits = torch.mean(pred_logits, dim=0).unsqueeze(0)
        if aux_logits is not None:
            aux_logits = aux_logits[0]
            aux_logits = torch.mean(aux_logits, dim=0)  # (q, c)
        outputs['pred_logits'] = out_logits
        outputs['ids'] = [torch.arange(0, outputs['pred_masks'].size(1))]
        if aux_logits is not None:
            return outputs, aux_logits
        return outputs