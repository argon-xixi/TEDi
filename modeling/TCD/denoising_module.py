import torch
from torch import nn
from modeling.transformer_decoder.mask2former_transformer_decoder import (
    CrossAttentionLayer,
    FFNLayer,
    MLP,
    SelfAttentionLayer,
    _get_activation_fn,
)
from scipy.optimize import linear_sum_assignment

class TCDCrossAttentionLayer(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
        normalize_before=False
    ):
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

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        indentify,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None
    ):
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]

        tgt = indentify + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self,
        indentify,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None
    ):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = indentify + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        indentify,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None
    ):

        if self.normalize_before:
            return self.forward_pre(indentify, tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(indentify, tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)

class TCDdenoising(torch.nn.Module):
    def __init__(
        self,
        hidden_channel=256,
        feedforward_channel=2048,
        num_head=8,
        decoder_layer_num=6,
        mask_dim=256,
        class_num=25,
    ):
        super(TCDdenoising, self).__init__()

        self.num_heads = num_head
        self.num_layers = decoder_layer_num
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.transformer_cross_attention_layers.append(
                TCDCrossAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_channel,
                    dim_feedforward=feedforward_channel,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_channel)

        self.class_embed = nn.Linear(hidden_channel, class_num + 1)
        self.mask_embed = MLP(hidden_channel, hidden_channel, mask_dim, 3)

        self.last_outputs = None
        self.last_frame_embeds = None

    def _clear_memory(self):
        del self.last_outputs
        self.last_outputs = None
        return

    def forward(self, frame_embeds, mask_features, resume=False, return_indices=False):

        frame_embeds = frame_embeds.permute(2, 3, 0, 1)
        n_frame, n_q, bs, _ = frame_embeds.size()
        outputs = []
        ret_indices = []

        for i in range(n_frame):
            ms_output = []
            single_frame_embeds = frame_embeds[i]

            if i == 0 and resume is False:
                self._clear_memory()
                self.last_frame_embeds = single_frame_embeds
                for j in range(self.num_layers):
                    if j == 0:
                        ms_output.append(single_frame_embeds)
                        ret_indices.append(self.match_embds(single_frame_embeds, single_frame_embeds))
                        output = self.transformer_cross_attention_layers[j](
                            single_frame_embeds, single_frame_embeds, single_frame_embeds,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )
                        output = self.transformer_self_attention_layers[j](
                            output, tgt_mask=None,
                            tgt_key_padding_mask=None,
                            query_pos=None
                        )

                        output = self.transformer_ffn_layers[j](
                            output
                        )
                        ms_output.append(output)
                    else:
                        output = self.transformer_cross_attention_layers[j](
                            ms_output[-1], ms_output[-1], single_frame_embeds,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )
                        output = self.transformer_self_attention_layers[j](
                            output, tgt_mask=None,
                            tgt_key_padding_mask=None,
                            query_pos=None
                        )

                        output = self.transformer_ffn_layers[j](
                            output
                        )
                        ms_output.append(output)
            else:
                for j in range(self.num_layers):
                    if j == 0:

                        ms_output.append(single_frame_embeds)
                        indices = self.match_embds(self.last_frame_embeds, single_frame_embeds)
                        self.last_frame_embeds = self.reorder_on_q_per_batch(single_frame_embeds,indices)
                        ret_indices.append(indices)

                        output = self.transformer_cross_attention_layers[j](
                            self.reorder_on_q_per_batch(single_frame_embeds,indices), self.last_outputs[-1], single_frame_embeds,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )
                        output = self.transformer_self_attention_layers[j](
                            output, tgt_mask=None,
                            tgt_key_padding_mask=None,
                            query_pos=None
                        )

                        output = self.transformer_ffn_layers[j](
                            output
                        )
                        ms_output.append(output)
                    else:
                        output = self.transformer_cross_attention_layers[j](
                            ms_output[-1], self.last_outputs[-1],single_frame_embeds,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )
                        output = self.transformer_self_attention_layers[j](
                            output, tgt_mask=None,
                            tgt_key_padding_mask=None,
                            query_pos=None
                        )

                        output = self.transformer_ffn_layers[j](
                            output
                        )
                        ms_output.append(output)
                        
            ms_output = torch.stack(ms_output, dim=0)
            self.last_outputs = ms_output
            outputs.append(ms_output[1:])
        outputs = torch.stack(outputs, dim=0)
        outputs_class, outputs_masks = self.prediction(outputs, mask_features)
        outputs = self.decoder_norm(outputs)
        out = {
           'pred_logits': outputs_class[-1].transpose(1, 2),
           'pred_masks': outputs_masks[-1],
           'aux_outputs': self._set_aux_loss(
               outputs_class, outputs_masks
           ),
           'pred_embds': outputs[:, -1].permute(2, 3, 0, 1)
        }
        if return_indices:
            return out, ret_indices
        else:
            return out

    def match_embds(self, ref_embds, cur_embds):

        q, b, c = ref_embds.shape

        ref = ref_embds.detach().permute(1, 0, 2).contiguous()
        cur = cur_embds.detach().permute(1, 0, 2).contiguous()

        perms = []
        for i in range(b):
            ref_i = ref[i]
            cur_i = cur[i]

            ref_i = ref_i / (ref_i.norm(dim=1, keepdim=True) + 1e-6)
            cur_i = cur_i / (cur_i.norm(dim=1, keepdim=True) + 1e-6)

            cos_sim = ref_i @ cur_i.t()
            C = (1 - cos_sim).cpu()
            C = torch.where(torch.isnan(C), torch.zeros_like(C), C)

            # Hungarian matching preserves query identity between adjacent frames.
            _, col_ind = linear_sum_assignment(C.t().numpy())
            perms.append(torch.from_numpy(col_ind).long())

        perm_bq = torch.stack(perms, dim=0).to(ref_embds.device)
        return perm_bq

    def reorder_on_q_per_batch(self,x_qbc, perm_bq):

        q, b, c = x_qbc.shape
        x_bqc = x_qbc.permute(1, 0, 2).contiguous()

        idx = perm_bq.unsqueeze(-1).expand(-1, -1, c)
        x_bqc_reordered = torch.gather(x_bqc, dim=1, index=idx)

        return x_bqc_reordered.permute(1, 0, 2).contiguous()

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):

        return [{"pred_logits": a.transpose(1, 2), "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
                ]

    def prediction(self, outputs, mask_features):

        decoder_output = self.decoder_norm(outputs)
        decoder_output = decoder_output.permute(1, 3, 0, 2, 4)
        outputs_class = self.class_embed(decoder_output).transpose(2, 3)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("lbtqc,btchw->lbqthw", mask_embed, mask_features)
        return outputs_class, outputs_mask

    def frame_forward(self, frame_embeds):

        bs, n_channel, n_frame, n_q = frame_embeds.size()
        frame_embeds = frame_embeds.permute(3, 0, 2, 1)
        frame_embeds = frame_embeds.flatten(1, 2)

        for j in range(self.num_layers):
            if j == 0:
                output = self.transformer_cross_attention_layers[j](
                    frame_embeds, frame_embeds, frame_embeds,
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=None, query_pos=None
                )
                output = self.transformer_self_attention_layers[j](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=None
                )

                output = self.transformer_ffn_layers[j](
                    output
                )
            else:
                output = self.transformer_cross_attention_layers[j](
                    output, output, frame_embeds,
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=None, query_pos=None
                )
                output = self.transformer_self_attention_layers[j](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=None
                )

                output = self.transformer_ffn_layers[j](
                    output
                )
        output = self.decoder_norm(output)
        output = output.reshape(n_q, bs, n_frame, n_channel)
        return output.permute(1, 3, 2, 0)


  