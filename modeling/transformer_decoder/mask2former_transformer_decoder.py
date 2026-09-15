

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

    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class MLP(nn.Module):

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

        query_variant: str = "original",
    ):
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

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

        self.query_feat = nn.Embedding(num_queries, hidden_dim)

        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()

        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        self.query_variant = query_variant

        if self.query_variant in ("topk", "linear"):

            self.stage_queries = [num_queries, 50, 10]
            self.num_stages = len(self.stage_queries)
            assert self.num_layers % self.num_stages == 0, "dec_layers 必须能被阶段数整除(例如 9 层配合 [100,50,10])"
            self.layers_per_stage = self.num_layers // self.num_stages
            assert (
                self.stage_queries[0] == num_queries
            ), "stage_queries[0] 必须等于 num_queries (初始 query 数量)"

            if self.query_variant == "linear":

                self.query_down_layers = nn.ModuleList([
                    nn.Linear(self.stage_queries[s], self.stage_queries[s + 1])
                    for s in range(self.num_stages - 1)
                ])
        else:
            self.stage_queries = None
            self.layers_per_stage = None

    def forward(self, x, mask_features,feat_sam = None, mask = None):

        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        query_embed = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        stage_outputs = []
        stage_indices = []
        if self.query_variant == "topk":

            current_indices = torch.arange(self.num_queries, device=output.device).unsqueeze(0).repeat(bs, 1)
        else:
            current_indices = None

        predictions_class = []
        predictions_mask = []

        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

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

            if (
                self.query_variant in ("topk", "linear")
                and (i + 1) % self.layers_per_stage == 0
            ):
                stage_outputs.append(output)
                if self.query_variant == "topk":
                    stage_indices.append(current_indices)

            if (
                self.query_variant in ("topk", "linear")
                and (i + 1) % self.layers_per_stage == 0
                and (i + 1) < self.num_layers
            ):

                finished_stage_idx = (i + 1) // self.layers_per_stage - 1
                next_stage_idx = finished_stage_idx + 1
                next_num_queries = self.stage_queries[next_stage_idx]

                next_level_index = (i + 1) % self.num_feature_levels
                next_attn_target_size = size_list[next_level_index]

                if self.query_variant == "topk":

                    cls_scores = outputs_class[..., :-1].max(-1).values

                    _, topk_idx = torch.topk(cls_scores, k=next_num_queries, dim=1)

                    out_bqc = output.permute(1, 0, 2)
                    qe_bqc = query_embed.permute(1, 0, 2)

                    B, Q, C = out_bqc.shape
                    _, _, H, W = outputs_mask.shape

                    idx_feat = topk_idx.unsqueeze(-1).expand(-1, -1, C)
                    idx_mask = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

                    out_bqc = torch.gather(out_bqc, 1, idx_feat)
                    qe_bqc = torch.gather(qe_bqc, 1, idx_feat)
                    outputs_mask = torch.gather(outputs_mask, 1, idx_mask)

                    current_indices = torch.gather(current_indices, 1, topk_idx)

                    output = out_bqc.permute(1, 0, 2)
                    query_embed = qe_bqc.permute(1, 0, 2)

                    attn_mask = self._build_attn_mask_from_outputs_mask(
                        outputs_mask,
                        attn_mask_target_size=next_attn_target_size,
                    )

                elif self.query_variant == "linear":

                    down_layer = self.query_down_layers[finished_stage_idx]

                    output_bcq = einops.rearrange(output, "q b c -> b c q")
                    output_bcq = down_layer(output_bcq)
                    output = einops.rearrange(output_bcq, "b c q -> q b c")

                    qe_bcq = einops.rearrange(query_embed, "q b c -> b c q")
                    qe_bcq = down_layer(qe_bcq)
                    query_embed = einops.rearrange(qe_bcq, "b c q -> q b c")

                    attn_mask = self._build_attn_mask_from_output(
                        output,
                        mask_features,
                        attn_mask_target_size=next_attn_target_size,
                    )

        assert len(predictions_class) == self.num_layers + 1

        if self.query_variant == "topk" and len(stage_outputs) > 0:

            C = output.shape[-1]

            feat_sum = output.new_zeros(bs, self.num_queries, C)
            count = output.new_zeros(bs, self.num_queries, 1)

            for stage_out, stage_idx in zip(stage_outputs, stage_indices):

                stage_out_bqc = stage_out.permute(1, 0, 2)

                idx_feat = stage_idx.unsqueeze(-1).expand(-1, -1, C)
                feat_sum.scatter_add_(dim=1, index=idx_feat, src=stage_out_bqc)

                ones = torch.ones(bs, stage_idx.shape[1], 1, device=output.device, dtype=output.dtype)
                idx_cnt = stage_idx.unsqueeze(-1).expand(-1, -1, 1)
                count.scatter_add_(dim=1, index=idx_cnt, src=ones)

            count = count.clamp_min(1.0)
            fused_bqc = feat_sum / count

            final_output = fused_bqc.permute(1, 0, 2)

            final_decoder_output = self.decoder_norm(final_output)
            final_decoder_output = final_decoder_output.transpose(0, 1)

            final_logits = self.class_embed(final_decoder_output)
            final_mask_embed = self.mask_embed(final_decoder_output)
            final_masks = torch.einsum("bqc,bchw->bqhw", final_mask_embed, mask_features)

            pred_embds = einops.rearrange(final_output, 'q b c -> b c q')
        else:

            norm_output = self.decoder_norm(output)
            pred_embds = einops.rearrange(norm_output, 'q b c -> b c q')
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

        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        attn_mask = self._build_attn_mask_from_outputs_mask(outputs_mask, attn_mask_target_size)

        return outputs_class, outputs_mask, attn_mask

    def forward_prediction_heads_new(self, output, mask_features, attn_mask_target_size):

        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        attn_mask = self._build_attn_mask_from_outputs_mask(outputs_mask, attn_mask_target_size)

        return outputs_class, outputs_mask, attn_mask

    def _build_attn_mask_from_outputs_mask(self, outputs_mask, attn_mask_target_size):

        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)

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
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        return self._build_attn_mask_from_outputs_mask(outputs_mask, attn_mask_target_size)

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):

        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
