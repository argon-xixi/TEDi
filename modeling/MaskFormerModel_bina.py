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
from torch import nn
from addict import Dict
import torch
from .backbone.resnet import ResNet, resnet_spec
from .backbone.swin import D2SwinTransformer
from .pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from .transformer_decoder.mask2former_transformer_decoder_sam import MultiScaleMaskedTransformerDecoder
from .transformer_decoder.mask2former_transformer_decoder_sam_bbox import MultiScaleMaskedTransformerDecoder_bbox
import sys
sys.path.append('/home/gjy/code_yjh/Mask2Former-Simplify-master/yjh')
from flow_warp import *

class MaskFormerHead_bina(nn.Module):
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
        # print(pixel_decoder.weight.device)
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
        bbox=cfg.bbox
        if bbox:
            predictor = MultiScaleMaskedTransformerDecoder_bbox(in_channels, 
                                                        num_classes, 
                                                        mask_classification,
                                                        bbox,
                                                        hidden_dim,
                                                        num_queries,
                                                        nheads,
                                                        dim_feedforward,
                                                        dec_layers,
                                                        pre_norm,
                                                        mask_dim,
                                                        enforce_input_project)
        else:
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

    def forward(self, features,feat_sam, mask=None):
        # 先融合再decoder
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)     #先经过 pixel_decoder再经过transformer_decoder
        predictions = self.predictor(multi_scale_features, mask_features, feat_sam,mask)        
        return predictions

class MaskFormerModel_bina(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = self.build_backbone(cfg)
        # self.backbone1 = self.build_backbone(cfg)
        # self.backbone2 = self.build_backbone(cfg)
        self.sem_seg_head = MaskFormerHead_bina(cfg, self.backbone_feature_shape)

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
            cfg.MODEL.SWIN.DEPTHS = swin_depth[cfg.MODEL.SWIN.TYPE]
            cfg.MODEL.SWIN.NUM_HEADS = swin_heads[cfg.MODEL.SWIN.TYPE]
            cfg.MODEL.SWIN.EMBED_DIM = swin_dim[cfg.MODEL.SWIN.TYPE]
            backbone = D2SwinTransformer(cfg)
            self.backbone_feature_shape = backbone.output_shape()
        else:
            raise NotImplementedError('Do not support model type!')
        return backbone

    def forward(self, inputs_left, inputs_right,feat_sam,flow_l2r):
        fused_features={}
        
        features_left = self.backbone(inputs_left) #输出不同尺寸的特征图（res2，res3，res4，res5）
        features_right = self.backbone(inputs_right)
        # print(flow_l2r.max())
        for k,v in features_right.items():
            feature_size=v.shape[2:]
            flow_resized=flow_resize(flow_l2r.permute(0, 3, 1, 2), feature_size)
            # print(flow_resized.max())
            feat_right_warped = warp(v, flow_resized)  # [B,C,H,W]
            similarity = compute_similarity(features_left[k], feat_right_warped)
            # occlusion_mask = (similarity < 0.5).float()  # 假设相似度低于0.5为遮挡
            # print(occlusion_mask.sum())
            fused_feature = fuse_features(features_left[k], feat_right_warped, similarity)
            # fused_feature = occlusion_mask * features_left[k] + (1 - occlusion_mask) * fused_feature
            # fused_feature = features_left[k] + occlusion_mask* fused_feature
            fused_features[k] = fused_feature
        # print('hah')
        
        outputs = self.sem_seg_head(fused_features,feat_sam)
        return outputs
            
