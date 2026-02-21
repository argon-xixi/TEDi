# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .position_encoding import PositionEmbeddingSine
import einops
class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MultiScaleMaskedTransformerDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        mask_classification=True,  
        hidden_dim=256,
        num_queries=100,
        nheads=8,
        dim_feedforward=2048,
        dec_layers=10,
        pre_norm=False,
        mask_dim=256,
        enforce_input_project=False,
        # query 数量调节策略："original"（原版）、"topk"（按得分剪枝）、"linear"（线性压缩）
        query_variant: str = "original",
    ):
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
        
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        #query做初始化
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # self.test = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        #计算query的pe
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        # ==============================
        #  query 数量调节相关参数
        # ==============================
        # query_variant: "original" | "topk" | "linear"
        self.query_variant = query_variant

        if self.query_variant in ("topk", "linear"):
            # 默认三阶段：100 -> 50 -> 10
            # 注意：要求 dec_layers 能被阶段数整除
            self.stage_queries = [num_queries, 50, 10]
            self.num_stages = len(self.stage_queries)
            assert self.num_layers % self.num_stages == 0, "dec_layers 必须能被阶段数整除(例如 9 层配合 [100,50,10])"
            self.layers_per_stage = self.num_layers // self.num_stages
            assert (
                self.stage_queries[0] == num_queries
            ), "stage_queries[0] 必须等于 num_queries (初始 query 数量)"

            # 线性压缩 Q（例如 100->50、50->10），在 Q 维做线性映射
            if self.query_variant == "linear":
                # 例如 [100, 50, 10] 会产生 2 个线性层：100->50, 50->10
                self.query_down_layers = nn.ModuleList([
                    nn.Linear(self.stage_queries[s], self.stage_queries[s + 1])
                    for s in range(self.num_stages - 1)
                ])
        else:
            self.stage_queries = None
            self.layers_per_stage = None

    def forward(self, x, mask_features,feat_sam = None, mask = None):
        # mask_feature
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2)) #位置编码
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None]) #将该特征图与 level_embed 中的权重相加（为每个特征图添加级别嵌入）。

            # flatten NxCxHxW to HWxNxCquery
            pos[-1] = pos[-1].permute(2, 0, 1) 
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1) #query位置编码
        output = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1) #初始化query特征

        # ==============================
        #  多阶段 topk 用于后续还原 & 融合
        #  - stage_outputs: 记录每个 stage 结束时的 output（[Q_s, B, C]）
        #  - stage_indices: 记录该 stage 的每个 query 在原始 num_queries 空间中的索引 [B, Q_s]
        # ==============================
        stage_outputs = []
        stage_indices = []
        if self.query_variant == "topk":
            # current_indices[b, q] = 该阶段第 q 个 query 对应的原始 query 索引（0~num_queries-1）
            current_indices = torch.arange(self.num_queries, device=output.device).unsqueeze(0).repeat(bs, 1)
        else:
            current_indices = None

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features  attn_mask为[B * num_heads, Q, target_height * target_width]
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            # 先计算交叉注意力
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )
            # 再计算自注意力
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads_new(
                output,
                mask_features,
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

            # 记录每个 stage 结束时的 output 及其在原始 query 空间中的索引
            # 注意：这里记录的是 stage 结束时、尚未进行下一阶段 topk/linear 压缩前的 output
            if (
                self.query_variant in ("topk", "linear")
                and (i + 1) % self.layers_per_stage == 0
            ):
                stage_outputs.append(output)
                if self.query_variant == "topk":
                    stage_indices.append(current_indices)

            # ==================================================
            #  在阶段结束处，根据策略调整 query 数量
            #  - original: 不做任何处理，保持固定 num_queries
            #  - topk:     按分类得分选择 top-k query 进入下一阶段
            #  - linear:   用线性层在 Q 维将 Q_old 压缩为 Q_new
            # ==================================================
            if (
                self.query_variant in ("topk", "linear")
                and (i + 1) % self.layers_per_stage == 0
                and (i + 1) < self.num_layers  # 最后一阶段结束后就不用再变换
            ):
                # 当前结束的是第几个阶段（0-based）
                finished_stage_idx = (i + 1) // self.layers_per_stage - 1
                next_stage_idx = finished_stage_idx + 1
                next_num_queries = self.stage_queries[next_stage_idx]

                # 下一层 cross-attn 会使用的特征图尺寸
                next_level_index = (i + 1) % self.num_feature_levels
                next_attn_target_size = size_list[next_level_index]

                if self.query_variant == "topk":
                    # ---------- 基于分类得分的 per-sample top-k 剪枝 ----------
                    # outputs_class: [B, Q, num_classes+1]
                    # 取除背景以外各类的最大得分，得到 [B, Q]
                    cls_scores = outputs_class[..., :-1].max(-1).values  # [B, Q]

                    # 对每个样本各自做 top-k（dim=1 是 Q 维）
                    # topk_idx: [B, next_Q]
                    _, topk_idx = torch.topk(cls_scores, k=next_num_queries, dim=1)

                    # 在 Q 维度上，按样本分别裁剪 output / query_embed / outputs_mask
                    # 当前：output, query_embed 形状为 [Q, B, C]，先转成 [B, Q, C]
                    out_bqc = output.permute(1, 0, 2)         # [B, Q, C]
                    qe_bqc = query_embed.permute(1, 0, 2)     # [B, Q, C]

                    B, Q, C = out_bqc.shape
                    _, _, H, W = outputs_mask.shape

                    # 扩展索引到特征维度，便于 gather
                    idx_feat = topk_idx.unsqueeze(-1).expand(-1, -1, C)           # [B, next_Q, C]
                    idx_mask = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)  # [B, next_Q, H, W]

                    out_bqc = torch.gather(out_bqc, 1, idx_feat)      # [B, next_Q, C]
                    qe_bqc = torch.gather(qe_bqc, 1, idx_feat)        # [B, next_Q, C]
                    outputs_mask = torch.gather(outputs_mask, 1, idx_mask)  # [B, next_Q, H, W]

                    # 更新当前阶段的索引映射：保留被选中的 query 的原始索引
                    # current_indices: [B, Q] -> [B, next_Q]
                    current_indices = torch.gather(current_indices, 1, topk_idx)

                    # 再转回 [Q, B, C]
                    output = out_bqc.permute(1, 0, 2)        # [next_Q, B, C]
                    query_embed = qe_bqc.permute(1, 0, 2)    # [next_Q, B, C]

                    # 根据裁剪后的 outputs_mask 重新计算下一阶段用的 attn_mask
                    attn_mask = self._build_attn_mask_from_outputs_mask(
                        outputs_mask,
                        attn_mask_target_size=next_attn_target_size,
                    )

                elif self.query_variant == "linear":
                    # ---------- 在线性层中在 Q 维做 100->50->10 的压缩 ----------
                    # output: [Q_old, B, C] -> [B, C, Q_old] -> 线性 -> [B, C, Q_new] -> [Q_new, B, C]
                    down_layer = self.query_down_layers[finished_stage_idx]

                    output_bcq = einops.rearrange(output, "q b c -> b c q")
                    output_bcq = down_layer(output_bcq)
                    output = einops.rearrange(output_bcq, "b c q -> q b c")

                    # 对当前的 query_embed（[Q_old, B, C]）在 Q 维做同样的线性变换
                    qe_bcq = einops.rearrange(query_embed, "q b c -> b c q")  # [B, C, Q_old]
                    qe_bcq = down_layer(qe_bcq)                                  # [B, C, Q_new]
                    query_embed = einops.rearrange(qe_bcq, "b c q -> q b c")    # [Q_new, B, C]

                    # 线性压缩后，直接通过新的 output + mask_features 重新计算 attn_mask
                    attn_mask = self._build_attn_mask_from_output(
                        output,
                        mask_features,
                        attn_mask_target_size=next_attn_target_size,
                    )
            
        
        assert len(predictions_class) == self.num_layers + 1

        # ==================================================
        #  最终预测：
        #  - 对于 topk：按照对应关系将各 stage 的 output 还原到原始 num_queries 维度并相加，
        #               然后用融合后的 100 个 query 做一次统一预测；
        #  - 其它变体：保持原有行为，直接使用最后一层的预测结果。
        # ==================================================

        if self.query_variant == "topk" and len(stage_outputs) > 0:
            # 按出现次数进行平均融合：
            # 对每个原始 query k，累加它在各 stage 的特征并统计出现次数，
            # 最终 h_fused[k] = feat_sum[k] / count[k]，避免“出现次数多的 query 数值被无脑放大”。

            C = output.shape[-1]
            # 在 [B, num_queries, C] 空间中累计特征和 & 出现次数
            feat_sum = output.new_zeros(bs, self.num_queries, C)      # [B, Q, C]
            count = output.new_zeros(bs, self.num_queries, 1)         # [B, Q, 1]

            for stage_out, stage_idx in zip(stage_outputs, stage_indices):
                # stage_out: [Q_s, B, C]
                # stage_idx: [B, Q_s]，每个 query 在原始 num_queries 空间中的索引
                stage_out_bqc = stage_out.permute(1, 0, 2)  # [B, Q_s, C]

                # -------- 特征累加 --------
                idx_feat = stage_idx.unsqueeze(-1).expand(-1, -1, C)  # [B, Q_s, C]
                feat_sum.scatter_add_(dim=1, index=idx_feat, src=stage_out_bqc)  # [B, Q, C]

                # -------- 出现次数累加 --------
                ones = torch.ones(bs, stage_idx.shape[1], 1, device=output.device, dtype=output.dtype)  # [B, Q_s, 1]
                idx_cnt = stage_idx.unsqueeze(-1).expand(-1, -1, 1)  # [B, Q_s, 1]
                count.scatter_add_(dim=1, index=idx_cnt, src=ones)   # [B, Q, 1]

            # 避免除 0（理论上第一阶段所有 query 都出现过，count>=1，这里只是防御性写法）
            count = count.clamp_min(1.0)
            fused_bqc = feat_sum / count  # [B, Q, C]

            final_output = fused_bqc.permute(1, 0, 2)  # [num_queries, B, C]

            final_decoder_output = self.decoder_norm(final_output)  # [num_queries, B, C]
            final_decoder_output = final_decoder_output.transpose(0, 1)  # [B, num_queries, C]

            final_logits = self.class_embed(final_decoder_output)  # [B, num_queries, num_classes+1]
            final_mask_embed = self.mask_embed(final_decoder_output)  # [B, num_queries, mask_dim]
            final_masks = torch.einsum("bqc,bchw->bqhw", final_mask_embed, mask_features)  # [B, num_queries, H, W]

            pred_embds = einops.rearrange(final_output, 'q b c -> b c q')  # [B, C, num_queries]
        else:
            # 保持原有行为：直接使用最后一层输出作为主预测
            norm_output = self.decoder_norm(output)  # [Q, B, C]
            pred_embds = einops.rearrange(norm_output, 'q b c -> b c q')  # [B, C, Q]
            final_logits = predictions_class[-1]
            final_masks = predictions_mask[-1]

        out = {
            'pred_embds': pred_embds,
            'mask_features' : mask_features,
            'pred_logits': final_logits,
            'pred_masks': final_masks,
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            )
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        #  mask_feature bchw不更新
        #  output -> outputs_class和mask_embed bqc
        # outputs_class表示类别信息 mask_embed和mask_features计算得到mask信息
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features) #mask_embed让mask_features加权求和

        # 使用统一的函数根据 outputs_mask 构造 attn_mask
        attn_mask = self._build_attn_mask_from_outputs_mask(outputs_mask, attn_mask_target_size)

        return outputs_class, outputs_mask, attn_mask
    
    def forward_prediction_heads_new(self, output, mask_features, attn_mask_target_size):
        #  mask_feature bchw不更新
        #  output -> outputs_class和mask_embed bqc
        # outputs_class表示类别信息 mask_embed和mask_features计算得到mask信息
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features) #mask_embed让mask_features加权求和

        # 使用统一的函数根据 outputs_mask 构造 attn_mask
        attn_mask = self._build_attn_mask_from_outputs_mask(outputs_mask, attn_mask_target_size)

        return outputs_class, outputs_mask, attn_mask


    def _build_attn_mask_from_outputs_mask(self, outputs_mask, attn_mask_target_size):
        """根据当前的 outputs_mask 构造 cross-attention 用的 attn_mask。

        outputs_mask: [B, Q, H, W]
        返回 attn_mask: [B * num_heads, Q, H' * W'] (bool)
        """
        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (
            attn_mask.sigmoid()
            .flatten(2)
            .unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1)
            < 0.5
        ).bool()
        return attn_mask.detach()


    def _build_attn_mask_from_output(self, output, mask_features, attn_mask_target_size):
        """在给定新的 output（可能已经线性压缩过 Q）和 mask_features 的情况下，重新计算 attn_mask。"""
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)  # [B, Q, C]
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        return self._build_attn_mask_from_outputs_mask(outputs_mask, attn_mask_target_size)


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
