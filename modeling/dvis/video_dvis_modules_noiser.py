import torch
from torch import nn
import sys
sys.path.append('/data/yjh_files/code/DVIS-main')
from mask2former_video.modeling.transformer_decoder.video_mask2former_transformer_decoder import SelfAttentionLayer,\
    CrossAttentionLayer, FFNLayer, MLP, _get_activation_fn
from scipy.optimize import linear_sum_assignment
import random
import numpy as np
import fvcore.nn.weight_init as weight_init


class Noiser:
    """Batch 版本 noiser，按 DVIS_Plus 的思路在 query 上施加噪声。

    注意：这里不重新做 ref-cur 的匹配，而是依赖外部传入的 base_perm_bq
    （由原来的 match_embds 计算得到），这样在“不开噪声”时可以完全保持
    原有逻辑不变，只是在开启噪声时对当前帧的 query 做额外扰动。
    """

    def __init__(self, noise_ratio=0.8, mode='none'):
        assert mode in ['none', 'rs', 'wa', 'cc']
        self.mode = mode
        self.noise_ratio = noise_ratio

    @staticmethod
    def reorder_on_q_per_batch(x_qbc, perm_bq):
        """与 ReferringTracker.reorder_on_q_per_batch 相同的重排逻辑。

        x_qbc:   (q, b, c)
        perm_bq: (b, q)
        返回:    (q, b, c)  每个 batch 按自己的 perm 在 q 维重排
        """
        q, b, c = x_qbc.shape
        x_bqc = x_qbc.permute(1, 0, 2).contiguous()              # (b, q, c)

        idx = perm_bq.unsqueeze(-1).expand(-1, -1, c)            # (b, q, c)
        x_bqc_reordered = torch.gather(x_bqc, dim=1, index=idx)  # (b, q, c)

        return x_bqc_reordered.permute(1, 0, 2).contiguous()     # (q, b, c)

    def _rs_noise_forward(self, cur_embeds):
        """Random Shuffle：每个 batch 在 q 维随机打乱。"""
        q, b, c = cur_embeds.shape

        perms = []
        noise_list = []
        for i in range(b):
            idx = torch.randperm(q, device=cur_embeds.device)  # (q,)
            perms.append(idx)
            noise_list.append(cur_embeds[:, i, :][idx])        # (q, c)

        perm_bq = torch.stack(perms, dim=0)          # (b, q)
        noise_init = torch.stack(noise_list, dim=1)  # (q, b, c)
        return perm_bq, noise_init

    def _wa_noise_forward(self, cur_embeds):
        """Weighted Average：按 DVIS_Plus 思路做 query 混合。"""
        q, b, c = cur_embeds.shape

        perms = []
        mixed_list = []
        for i in range(b):
            idx = torch.randperm(q, device=cur_embeds.device)
            shuffled = cur_embeds[:, i, :][idx]          # (q, c)
            w = torch.rand(q, 1, device=cur_embeds.device)  # (q, 1)
            mixed = cur_embeds[:, i, :] * w + shuffled * (1.0 - w)

            ret_idx = torch.arange(q, device=cur_embeds.device)
            mask = (w.squeeze(-1) < 0.5)
            ret_idx[mask] = idx[mask]

            perms.append(ret_idx)
            mixed_list.append(mixed)

        perm_bq = torch.stack(perms, dim=0)           # (b, q)
        noise_init = torch.stack(mixed_list, dim=1)   # (q, b, c)
        return perm_bq, noise_init

    def _cc_noise_forward(self, cur_embeds):
        """Channel Cut：在通道维拼接原特征和乱序特征。"""
        q, b, c = cur_embeds.shape

        perms = []
        mixed_list = []
        for i in range(b):
            split = torch.randint(0, c, (q,), device=cur_embeds.device)  # (q,)
            chan_idx = torch.arange(c, device=cur_embeds.device).unsqueeze(0)  # (1, c)
            weight = (chan_idx < split.unsqueeze(-1)).float()                  # (q, c)

            idx = torch.randperm(q, device=cur_embeds.device)
            shuffled = cur_embeds[:, i, :][idx]  # (q, c)

            mixed = cur_embeds[:, i, :] * weight + shuffled * (1.0 - weight)

            ret_idx = torch.arange(q, device=cur_embeds.device)
            mask = (split < c // 2)
            ret_idx[mask] = idx[mask]

            perms.append(ret_idx)
            mixed_list.append(mixed)

        perm_bq = torch.stack(perms, dim=0)           # (b, q)
        noise_init = torch.stack(mixed_list, dim=1)   # (q, b, c)
        return perm_bq, noise_init

    def __call__(self, cur_embeds, base_perm_bq, activate=False):
        """根据基础匹配 perm_bq，在当前帧 query 上可选地加噪声。

        cur_embeds:   (q, b, c)  当前帧的 query 特征
        base_perm_bq: (b, q)     由原 match_embds 得到的匹配置换
        activate:     是否允许启用噪声（一般设为 self.training）

        返回:
            perm_bq:    实际使用的置换 (b, q)
            noise_init: 作为第一层 cross-attn query 的特征 (q, b, c)
        """
        # 先按原来的匹配结果做一次重排，对应“无噪声”情形
        aligned = self.reorder_on_q_per_batch(cur_embeds, base_perm_bq)

        # 不激活噪声 / 模式为 none / 或随机未命中时，直接返回原逻辑
        if (not activate) or self.mode == 'none' or (random.random() >= self.noise_ratio):
            return base_perm_bq, aligned

        # 启用噪声：参考 DVIS_Plus 的做法，忽略 base_perm，直接在当前帧上加噪
        if self.mode == 'rs':
            return self._rs_noise_forward(cur_embeds)
        elif self.mode == 'wa':
            return self._wa_noise_forward(cur_embeds)
        elif self.mode == 'cc':
            return self._cc_noise_forward(cur_embeds)
        else:
            raise NotImplementedError


class ReferringCrossAttentionLayer(nn.Module):

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
        # print(indentify.shape)
       
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
        # when set "indentify = tgt", ReferringCrossAttentionLayer is same as CrossAttentionLayer
        if self.normalize_before:
            return self.forward_pre(indentify, tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(indentify, tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)

class ReferringTracker(torch.nn.Module):
    def __init__(
        self,
        hidden_channel=256,
        feedforward_channel=2048,
        num_head=8,
        decoder_layer_num=6,
        mask_dim=256,
        class_num=25,
        noise_mode='none',
        noise_ratio=0.0,
        use_ref_proj=False,
    ):
        super(ReferringTracker, self).__init__()

        # init transformer layers
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
                ReferringCrossAttentionLayer(
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

        # whether to use ref_proj branch (similar to DVIS_Plus)
        self.use_ref_proj = use_ref_proj

        # projection head for reference branch
        if self.use_ref_proj:
            self.ref_proj = MLP(hidden_channel, hidden_channel, hidden_channel, 3)
            for layer in self.ref_proj.layers:
                weight_init.c2_xavier_fill(layer)

        # init heads (cls head input dim depends on whether ref_proj is used)
        cls_in_dim = 2 * hidden_channel if self.use_ref_proj else hidden_channel
        self.class_embed = nn.Linear(cls_in_dim, class_num + 1)
        self.mask_embed = MLP(hidden_channel, hidden_channel, mask_dim, 3)

        # record previous frame information
        self.last_outputs = None
        self.last_frame_embeds = None

        # noiser：在保持原匹配逻辑的基础上，对当前帧的 query 施加噪声
        self.noiser = Noiser(noise_ratio=noise_ratio, mode=noise_mode)

    def _clear_memory(self):
        del self.last_outputs
        self.last_outputs = None
        return

    def forward(self, frame_embeds, mask_features, resume=False, return_indices=False):
        """
        :param frame_embeds: the instance queries output by the segmenter
        :param mask_features: the mask features output by the segmenter
        :param resume: whether the first frame is the start of the video
        :param return_indices: whether return the match indices
        :return: output dict, including masks, classes, embeds.
        """
        frame_embeds = frame_embeds.permute(2, 3, 0, 1)  # t, q, b, c
        n_frame, n_q, bs, _ = frame_embeds.size()
        outputs = []
        ret_indices = []

        for i in range(n_frame):
            ms_output = []
            single_frame_embeds = frame_embeds[i]  # q b c
            # the first frame of a video
            if i == 0 and resume is False:
                self._clear_memory()
                self.last_frame_embeds = single_frame_embeds
                for j in range(self.num_layers):
                    if j == 0:
                        ms_output.append(single_frame_embeds)
                        ret_indices.append(self.match_embds(single_frame_embeds, single_frame_embeds))
                        if self.use_ref_proj:
                            # 第一帧第 0 层：用 ref_proj(single_frame_embeds) 作为 Q，memory 仍然是当前帧
                            q_feat = self.ref_proj(single_frame_embeds)
                            output = self.transformer_cross_attention_layers[j](
                                single_frame_embeds, q_feat, single_frame_embeds,
                                memory_mask=None,
                                memory_key_padding_mask=None,
                                pos=None, query_pos=None
                            )
                        else:
                            # 保持原始逻辑
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
                        # FFN
                        output = self.transformer_ffn_layers[j](
                            output
                        )
                        ms_output.append(output)
                    else:
                        if self.use_ref_proj:
                            # 第一帧后续层：用 ref_proj(上一层输出) 作为 Q
                            q_feat = self.ref_proj(ms_output[-1])
                            output = self.transformer_cross_attention_layers[j](
                                ms_output[-1], q_feat, single_frame_embeds,
                                memory_mask=None,
                                memory_key_padding_mask=None,
                                pos=None, query_pos=None
                            )
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
                        # FFN
                        output = self.transformer_ffn_layers[j](
                            output
                        )
                        ms_output.append(output)
            else:
                # 对于第 2 帧及之后，若使用 ref_proj，则先根据上一帧的最后一层输出计算 reference
                if self.use_ref_proj:
                    # self.last_outputs: (1 + L, q, b, c)  ->  取最后一层 (q, b, c)
                    reference = self.ref_proj(self.last_outputs[-1])

                for j in range(self.num_layers):
                    if j == 0:
                        # 第二帧及之后：先按原逻辑做匹配，然后在此基础上可选地加噪声
                        ms_output.append(single_frame_embeds)
                        base_indices = self.match_embds(self.last_frame_embeds, single_frame_embeds)  # (b, q)

                        # 使用 noiser 对当前帧 query 做扰动；在 eval 阶段或 noise_ratio=0 时退化为原逻辑
                        perm_bq, noised_init = self.noiser(
                            single_frame_embeds,
                            base_indices,
                            activate=self.training,
                        )

                        # 记录并更新上一帧特征（按实际使用的 perm 对齐）
                        self.last_frame_embeds = self.reorder_on_q_per_batch(single_frame_embeds, perm_bq)
                        ret_indices.append(perm_bq)
                        # 第 0 层：若开启 ref_proj，用 reference 作为 Q；否则保持原逻辑
                        if self.use_ref_proj:
                            tgt_feat = reference
                        else:
                            tgt_feat = self.last_outputs[-1]

                        output = self.transformer_cross_attention_layers[j](
                            noised_init, tgt_feat, single_frame_embeds,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )
                        output = self.transformer_self_attention_layers[j](
                            output, tgt_mask=None,
                            tgt_key_padding_mask=None,
                            query_pos=None
                        )
                        # FFN
                        output = self.transformer_ffn_layers[j](
                            output
                        )
                        ms_output.append(output)
                    else:
                        # 后续层：若开启 ref_proj，始终用上一帧 reference 作为 Q
                        if self.use_ref_proj:
                            tgt_feat = reference
                        else:
                            tgt_feat = self.last_outputs[-1]

                        output = self.transformer_cross_attention_layers[j](
                            ms_output[-1], tgt_feat, single_frame_embeds,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )
                        output = self.transformer_self_attention_layers[j](
                            output, tgt_mask=None,
                            tgt_key_padding_mask=None,
                            query_pos=None
                        )
                        # FFN
                        output = self.transformer_ffn_layers[j](
                            output
                        )
                        ms_output.append(output)
            ms_output = torch.stack(ms_output, dim=0)  # (1 + layers, q, b, c)
            self.last_outputs = ms_output
            outputs.append(ms_output[1:])
        outputs = torch.stack(outputs, dim=0)  # (t, l, q, b, c)
        outputs_class, outputs_masks = self.prediction(outputs, mask_features)
        outputs = self.decoder_norm(outputs)
        out = {
           'pred_logits': outputs_class[-1].transpose(1, 2),  # (b, t, q, c)
           'pred_masks': outputs_masks[-1],  # (b, q, t, h, w)
           'aux_outputs': self._set_aux_loss(
               outputs_class, outputs_masks
           ),
           'pred_embds': outputs[:, -1].permute(2, 3, 0, 1)  # (b, c, t, q)
        }
        if return_indices:
            return out, ret_indices
        else:
            return out

    def match_embds(self, ref_embds, cur_embds):
        """
        ref_embds, cur_embds: (q, b, c)
        return:
        perm_bq: (b, q)  每个 batch 一条长度为 q 的置换
        """
        q, b, c = ref_embds.shape

        ref = ref_embds.detach().permute(1, 0, 2).contiguous()  # (b,q,c)
        cur = cur_embds.detach().permute(1, 0, 2).contiguous()  # (b,q,c)

        perms = []
        for i in range(b):
            ref_i = ref[i]  # (q,c)
            cur_i = cur[i]  # (q,c)

            ref_i = ref_i / (ref_i.norm(dim=1, keepdim=True) + 1e-6)
            cur_i = cur_i / (cur_i.norm(dim=1, keepdim=True) + 1e-6)

            cos_sim = ref_i @ cur_i.t()     # (q,q)
            C = (1 - cos_sim).cpu()
            C = torch.where(torch.isnan(C), torch.zeros_like(C), C)

            # 得到长度 q 的列索引置换
            _, col_ind = linear_sum_assignment(C.t().numpy())
            perms.append(torch.from_numpy(col_ind).long())

        perm_bq = torch.stack(perms, dim=0).to(ref_embds.device)  # (b,q)
        return perm_bq


    def reorder_on_q_per_batch(self,x_qbc, perm_bq):
        """
        x_qbc:   (q, b, c)
        perm_bq: (b, q)
        return:  (q, b, c)  每个 batch 按自己的 perm 在 q 维重排
        """
        q, b, c = x_qbc.shape
        x_bqc = x_qbc.permute(1, 0, 2).contiguous()              # (b,q,c)

        idx = perm_bq.unsqueeze(-1).expand(-1, -1, c)            # (b,q,c)
        x_bqc_reordered = torch.gather(x_bqc, dim=1, index=idx)  # (b,q,c)

        return x_bqc_reordered.permute(1, 0, 2).contiguous()     # (q,b,c)

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a.transpose(1, 2), "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
                ]

    def prediction(self, outputs, mask_features):
        # outputs (t, l, q, b, c)
        # mask_features (b, t, c, h, w)
        decoder_output = self.decoder_norm(outputs)
        decoder_output = decoder_output.permute(1, 3, 0, 2, 4)  # (l, b, t, q, c)

        # 若开启 ref_proj，则在分类分支中拼接 reference 特征
        if self.use_ref_proj:
            ref_feat = self.ref_proj(decoder_output)  # (l, b, t, q, c)
            decoder_output_cls = torch.cat([ref_feat, decoder_output], dim=-1)
        else:
            decoder_output_cls = decoder_output

        outputs_class = self.class_embed(decoder_output_cls).transpose(2, 3)  # (l, b, q, t, cls+1)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("lbtqc,btchw->lbqthw", mask_embed, mask_features)
        return outputs_class, outputs_mask

    def frame_forward(self, frame_embeds):
        """
        only for providing the instance memories for refiner
        :param frame_embeds: the instance queries output by the segmenter, shape is (q, b, t, c)
        :return: the projected instance queries
        """
        bs, n_channel, n_frame, n_q = frame_embeds.size()
        frame_embeds = frame_embeds.permute(3, 0, 2, 1)  # (q, b, t, c)
        frame_embeds = frame_embeds.flatten(1, 2)  # (q, bt, c)

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
                # FFN
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
                # FFN
                output = self.transformer_ffn_layers[j](
                    output
                )
        output = self.decoder_norm(output)
        output = output.reshape(n_q, bs, n_frame, n_channel)
        return output.permute(1, 3, 2, 0)


class TemporalRefiner(torch.nn.Module):
    def __init__(
        self,
        hidden_channel=256,
        feedforward_channel=2048,
        num_head=8,
        decoder_layer_num=6,
        mask_dim=256,
        class_num=25,
        windows=5
    ):
        super(TemporalRefiner, self).__init__()

        self.windows = windows

        # init transformer layers
        self.num_heads = num_head
        self.num_layers = decoder_layer_num
        self.transformer_obj_self_attention_layers = nn.ModuleList()
        self.transformer_time_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        self.conv_short_aggregate_layers = nn.ModuleList()
        self.conv_norms = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_time_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.conv_short_aggregate_layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_channel, hidden_channel,
                              kernel_size=5, stride=1,
                              padding='same', padding_mode='replicate'),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(hidden_channel, hidden_channel,
                              kernel_size=3, stride=1,
                              padding='same', padding_mode='replicate'),
                )
            )

            self.conv_norms.append(nn.LayerNorm(hidden_channel))

            self.transformer_obj_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
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

        # init heads
        self.class_embed = nn.Linear(hidden_channel, class_num + 1)
        self.mask_embed = MLP(hidden_channel, hidden_channel, mask_dim, 3)

        self.activation_proj = nn.Linear(hidden_channel, 1)

    def forward(self, instance_embeds, frame_embeds, mask_features):
        """
        :param instance_embeds: the aligned instance queries output by the tracker, shape is (b, c, t, q)
        :param frame_embeds: the instance queries processed by the tracker.frame_forward function, shape is (b, c, t, q)
        :param mask_features: the mask features output by the segmenter, shape is (b, t, c, h, w)
        :return: output dict, including masks, classes, embeds.
        """
        n_batch, n_channel, n_frames, n_instance = instance_embeds.size()

        outputs = []
        output = instance_embeds
        frame_embeds = frame_embeds.permute(3, 0, 2, 1).flatten(1, 2)

        for i in range(self.num_layers):
            output = output.permute(2, 0, 3, 1)  # (t, b, q, c)
            output = output.flatten(1, 2)  # (t, bq, c)

            # do long temporal attention
            output = self.transformer_time_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=None
            )

            # do short temporal conv
            output = output.permute(1, 2, 0)  # (bq, c, t)
            output = self.conv_norms[i](
                (self.conv_short_aggregate_layers[i](output) + output).transpose(1, 2)
            ).transpose(1, 2)
            output = output.reshape(
                n_batch, n_instance, n_channel, n_frames
            ).permute(1, 0, 3, 2).flatten(1, 2)  # (q, bt, c)

            # do objects self attention
            output = self.transformer_obj_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=None
            )

            # do cross attention
            output = self.transformer_cross_attention_layers[i](
                output, frame_embeds,
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=None, query_pos=None
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            output = output.reshape(n_instance, n_batch, n_frames, n_channel).permute(1, 3, 2, 0)  # (b, c, t, q)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=0).permute(3, 0, 4, 1, 2)  # (l, b, c, t, q) -> (t, l, q, b, c)
        outputs_class, outputs_masks = self.prediction(outputs, mask_features)
        outputs = self.decoder_norm(outputs)
        out = {
           'pred_logits': outputs_class[-1].transpose(1, 2),  # (b, t, q, c)
           'pred_masks': outputs_masks[-1],  # (b, q, t, h, w)
           'aux_outputs': self._set_aux_loss(
               outputs_class, outputs_masks
           ),
           'pred_embds': outputs[:, -1].permute(2, 3, 0, 1)  # (b, c, t, q)
        }
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a.transpose(1, 2), "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
                ]

    def windows_prediction(self, outputs, mask_features, windows=5):
        """
        for windows prediction, because mask features consumed too much GPU memory
        """
        iters = outputs.size(0) // windows
        if outputs.size(0) % windows != 0:
            iters += 1
        outputs_classes = []
        outputs_masks = []
        for i in range(iters):
            start_idx = i * windows
            end_idx = (i + 1) * windows
            clip_outputs = outputs[start_idx:end_idx]
            decoder_output = self.decoder_norm(clip_outputs)
            decoder_output = decoder_output.permute(1, 3, 0, 2, 4)  # (l, b, t, q, c)
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum(
                "lbtqc,btchw->lbqthw",
                mask_embed,
                mask_features[:, start_idx:end_idx].to(mask_embed.device)
            )
            outputs_classes.append(decoder_output)
            outputs_masks.append(outputs_mask.cpu().to(torch.float32))
        outputs_classes = torch.cat(outputs_classes, dim=2)
        outputs_classes = self.pred_class(outputs_classes)
        return outputs_classes.cpu().to(torch.float32), torch.cat(outputs_masks, dim=3)

    def pred_class(self, decoder_output):
        """
        fuse the objects queries of all frames and predict an overall score based on the fused objects queries
        :param decoder_output: instance queries, shape is (l, b, t, q, c)
        """
        T = decoder_output.size(2)

        # compute the weighted average of the decoder_output
        activation = self.activation_proj(decoder_output).softmax(dim=2)  # (l, b, t, q, 1)
        class_output = (decoder_output * activation).sum(dim=2, keepdim=True)  # (l, b, 1, q, c)

        # to unify the output format, duplicate the fused features T times
        class_output = class_output.repeat(1, 1, T, 1, 1)
        outputs_class = self.class_embed(class_output).transpose(2, 3)
        return outputs_class

    def prediction(self, outputs, mask_features):
        """
        :param outputs: instance queries, shape is (t, l, q, b, c)
        :param mask_features: mask features, shape is (b, t, c, h, w)
        :return: pred class and pred masks
        """
        if self.training:
            decoder_output = self.decoder_norm(outputs)
            decoder_output = decoder_output.permute(1, 3, 0, 2, 4)  # (l, b, t, q, c)
            outputs_class = self.pred_class(decoder_output)
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("lbtqc,btchw->lbqthw", mask_embed, mask_features)
        else:
            outputs_class, outputs_mask = self.windows_prediction(outputs, mask_features, windows=self.windows)
        return outputs_class, outputs_mask
