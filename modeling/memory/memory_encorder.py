import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Optional, Dict


def maskscore(sel_mask_logits: torch.Tensor, pred_masks_bin: Optional[torch.Tensor] = None, mask_bin_thresh: float = 0.0) -> torch.Tensor:
    """简单的 mask 质量评估函数，与 classwise 版本保持一致。

    Args:
        sel_mask_logits: (N, H, W)，通常是某一帧中 B*Q 个 mask 的 logit
        pred_masks_bin:  这里不使用，保留接口兼容性
        mask_bin_thresh: 二值化阈值

    Returns:
        mask_scores: (N,) 每个 mask 的质量分数
    """

    pred_masks_bin = (sel_mask_logits > mask_bin_thresh).float()  # [N, H, W]
    mask_prob = sel_mask_logits.sigmoid()                         # [N, H, W]
    numer = (mask_prob.flatten(1) * pred_masks_bin.flatten(1)).sum(1)  # [N]
    denom = pred_masks_bin.flatten(1).sum(1).clamp_min(1e-6)          # [N]
    mask_scores = numer / denom                                      # [N]
    return mask_scores

class MaskTokenEncoder(nn.Module):
    """
    Encode per-query mask (H,W) into a C-dim token.
    Input:  masks (B, Q, T, H, W)
    Output: mask_tokens (B, T, Q, C)
    """
    def __init__(self, C: int, down_h: int = 32, down_w: int = 32):
        super().__init__()
        self.down_h = down_h
        self.down_w = down_w
        self.proj = nn.Linear(down_h * down_w, C)

    def forward(self, masks_bqthw: torch.Tensor) -> torch.Tensor:
        B, Q, T, H, W = masks_bqthw.shape

        # (B,Q,T,H,W) -> (B,T,Q,H,W)
        x = masks_bqthw.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * T * Q, 1, H, W)

        # stabilize values (optional but recommended)
        # if masks are logits, tanh keeps sign and bounds magnitude
        x = torch.tanh(x)

        x = F.interpolate(
            x,
            size=(self.down_h, self.down_w),
            mode="bilinear",
            align_corners=False,
        )
        x = x.flatten(1)               # (B*T*Q, down_h*down_w)
        x = self.proj(x)               # (B*T*Q, C)
        x = x.view(B, T, Q, -1)        # (B, T, Q, C)
        return x


class TemporalEncoding(nn.Module):
    """
    Absolute frame-id embedding. Adds (1, T, 1, C) to tokens shaped (B, T, Q, C).
    """
    def __init__(self, C: int, max_frames: int = 1024):
        super().__init__()
        self.time_emb = nn.Embedding(max_frames, C)

    def forward(self, x_btqc: torch.Tensor, frame_ids_t: torch.Tensor) -> torch.Tensor:
        """
        x_btqc: (B, T, Q, C)
        frame_ids_t: (T,) long, absolute frame indices for these T positions
        """
        assert frame_ids_t.dim() == 1
        T = x_btqc.shape[1]
        assert frame_ids_t.numel() == T

        e = self.time_emb(frame_ids_t).view(1, T, 1, -1)  # (1,T,1,C)
        return x_btqc + e


class GatedFuser(nn.Module):
    """
    Gated fusion of query token and mask token for each (t,q).
    """
    def __init__(self, C: int, hidden: int = 1024, drop: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(2 * C)
        self.gate = nn.Sequential(
            nn.Linear(2 * C, C),
            nn.Sigmoid()
        )
        self.mlp = nn.Sequential(
            nn.Linear(2 * C, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, C),
            nn.Dropout(drop),
        )

    def forward(self, q_btqc: torch.Tensor, m_btqc: torch.Tensor) -> torch.Tensor:
        z = torch.cat([q_btqc, m_btqc], dim=-1)  # (B,T,Q,2C)
        z = self.norm(z)
        g = self.gate(z)                         # (B,T,Q,C)
        d = self.mlp(z)                          # (B,T,Q,C)
        return q_btqc + g * d


class QueryMemoryLayer(nn.Module):
    """一层 SAM-style 的 query-level memory attention: SA + CA + MLP。

    输入：
      - q_bqc:   (B, Q, C)  当前帧（或当前去噪后的）query tokens
      - mem_bLc: (B, L_mem, C) 来自 memory bank 的 tokens（多帧拼接）
    输出：
      - (B, Q, C) 同形状的融合后 tokens
    """

    def __init__(self, C: int, num_heads: int = 8, dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        # 自注意力：只在当前 Q 个 query 之间建关系
        self.self_attn = nn.MultiheadAttention(
            embed_dim=C,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # 跨注意力：当前 Q 个 query 去看 memory bank 里的所有 tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=C,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # 前馈网络
        self.linear1 = nn.Linear(C, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, C)

        self.norm1 = nn.LayerNorm(C)
        self.norm2 = nn.LayerNorm(C)
        self.norm3 = nn.LayerNorm(C)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, q_bqc: torch.Tensor, mem_bLc: torch.Tensor) -> torch.Tensor:
        """q_bqc: (B,Q,C)，mem_bLc: (B,L_mem,C)"""
        # 1) Self-Attention on q
        x = self.norm1(q_bqc)
        x, _ = self.self_attn(x, x, x, need_weights=False)  # (B,Q,C)
        q_bqc = q_bqc + self.dropout1(x)

        # 2) Cross-Attention: q attends to memory
        x = self.norm2(q_bqc)
        x, _ = self.cross_attn(x, mem_bLc, mem_bLc, need_weights=False)  # (B,Q,C)
        q_bqc = q_bqc + self.dropout2(x)

        # 3) FFN
        x = self.norm3(q_bqc)
        x = self.linear2(self.dropout3(self.activation(self.linear1(x))))  # (B,Q,C)
        q_bqc = q_bqc + x
        return q_bqc


class VideoQueryMemoryModule(nn.Module):
    """
    Query-level memory module for video frames.

    Typical usage pattern:
      - feed a clip with T frames
      - use the first `mem_frames` frames to initialize the memory bank
      - then process a list of frames (e.g. last N frames) sequentially:
          write current -> read memory -> output fused tokens

    Inputs:
      queries: (B, C, T, Q)
      masks:   (B, Q, T, H, W)
      query_denoisy: (B, C, N, Q) where N = len(proc_frame_ids)

    Outputs:
      fused_out: (B, C, N, Q)  # aligned with proc_frame_ids
    """
    def __init__(
        self,
        C: int = 256,
        Q: int = 10,
        mem_frames: int = 3,
        fuser_hidden: int = 1024,
        down_h: int = 32,
        down_w: int = 32,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_frames: int = 1024,
        detach_memory: bool = False,
        num_layers: int = 2,
        topk: Optional[int] = None,
    ):
        super().__init__()
        self.C = C
        self.Q = Q
        self.mem_frames = mem_frames
        self.detach_memory = detach_memory
        # 若 topk 为 None，则保持原始 FIFO 行为；否则每帧只写入分数最高的 topk 个 query
        self.topk = topk

        self.mask_encoder = MaskTokenEncoder(C=C, down_h=down_h, down_w=down_w)
        self.temporal = TemporalEncoding(C=C, max_frames=max_frames)
        self.fuser = GatedFuser(C=C, hidden=fuser_hidden, drop=dropout)

        # 多层 QueryMemoryLayer：每层 = self-attn + cross-attn + MLP
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            QueryMemoryLayer(C=C, num_heads=num_heads, dim_feedforward=fuser_hidden, dropout=dropout)
            for _ in range(self.num_layers)
        ])

        # FIFO memory bank stores per-frame tokens shaped (B, Q, C) with known absolute frame id
        self._memory_bank = {}  # device -> deque(maxlen=mem_frames)

    def _get_bank(self, device):
        if device not in self._memory_bank:
            self._memory_bank[device] = deque(maxlen=self.mem_frames)
        return self._memory_bank[device]


    def reset_memory(self):
        self._memory_bank.clear()

    def _encode_frame_tokens(
        self,
        q_bctq: torch.Tensor,        # (B,C,1,Q) or (B,C,Q) reshaped inside
        m_bqthw: torch.Tensor,        # (B,Q,1,H,W)
        frame_id: int,
    ) -> torch.Tensor:
        """
        Returns tokens (B, Q, C) for a single frame (with temporal encoding + gated fusion).
        """
        # query -> (B,1,Q,C)
        if q_bctq.dim() == 4:
            # (B,C,1,Q) -> (B,1,Q,C)
            q_btqc = q_bctq.permute(0, 2, 3, 1).contiguous()
        elif q_bctq.dim() == 3:
            # (B,C,Q) -> (B,1,Q,C)
            q_btqc = q_bctq.permute(0, 2, 1).unsqueeze(1).contiguous()
        else:
            raise ValueError("q_bctq must be (B,C,1,Q) or (B,C,Q)")

        # mask -> (B,1,Q,C)
        m_btqc = self.mask_encoder(m_bqthw)  # (B,1,Q,C)

        # temporal encoding (absolute frame id)
        frame_ids = torch.tensor([frame_id], device=q_bctq.device, dtype=torch.long)  # (1,)
        q_btqc = self.temporal(q_btqc, frame_ids)
        m_btqc = self.temporal(m_btqc, frame_ids)

        # gated fusion -> (B,1,Q,C)
        fused_btqc = self.fuser(q_btqc, m_btqc)

        # return (B,Q,C)
        return fused_btqc.squeeze(1)

    def _read_memory(self, cur_tokens_bqc: torch.Tensor) -> torch.Tensor:
        """多层 memory attention：当前 tokens (B,Q,C) 对 memory (B,mem_len,C)。"""
        dev = cur_tokens_bqc.device
        bank = self._get_bank(dev)
        if len(bank) == 0:
            return cur_tokens_bqc
        mem_tokens = torch.cat([item["tokens"] for item in bank], dim=1)  # (B,mem_frames*Q,C)
        x = cur_tokens_bqc
        for layer in self.layers:
            x = layer(x, mem_tokens)
        return x

    def _write_memory(
        self,
        tokens_bqc: torch.Tensor,
        frame_id: int,
        scores_bq: Optional[torch.Tensor] = None,
        topk: Optional[int] = None,
    ):
        if self.detach_memory:
            tokens_bqc = tokens_bqc.detach()
        dev = tokens_bqc.device
        bank = self._get_bank(dev)
        # 若提供了 scores_bq 且指定了 topk，则只写入分数最高的 topk 个 query
        if (scores_bq is not None) and (topk is not None):
            B, Q, C = tokens_bqc.shape
            assert scores_bq.shape[:2] == (B, Q), f"scores_bq shape {scores_bq.shape} mismatch with tokens {tokens_bqc.shape}"
            k = min(topk, Q)
            # (B, k)
            topk_scores, topk_idx = torch.topk(scores_bq, k=k, dim=1)
            batch_idx = torch.arange(B, device=dev).unsqueeze(1).expand(-1, k)  # (B,k)
            # (B,k,C)
            tokens_topk = tokens_bqc[batch_idx, topk_idx, :]
            bank.append({"tokens": tokens_topk, "frame_id": frame_id, "scores": topk_scores})
        else:
            # 兼容原始 FIFO 行为：不做筛选，整帧写入
            bank.append({"tokens": tokens_bqc, "frame_id": frame_id})

    def _compute_final_scores(
        self,
        logits_bqc: torch.Tensor,   # (B, Q, num_classes+1)
        m_bq1hw: torch.Tensor,      # (B, Q, 1, H, W)
    ) -> torch.Tensor:
        """按照 "分类置信度 * maskscore" 计算每个 query 的最终得分。

        参考 MaskFormerModel_track_memorybank_classwise 中的实现。
        """
        # 先通过 softmax 得到每个 query 的类别概率
        probs_btqc = torch.softmax(logits_bqc, dim=-1)[:, :, :-1]  # (B, Q, num_classes)
        max_probs_bq, _ = probs_btqc.max(dim=-1)                   # (B, Q)

        # 进一步结合当前帧 mask 的质量分数（maskscore）作为额外置信度
        # sel_mask_logits: (B,Q,H,W)，与该帧的 logits_bqc 对应
        sel_mask_logits = m_bq1hw[:, :, 0, :, :]                   # (B, Q, H, W)
        Bm, Qm, Hm, Wm = sel_mask_logits.shape

        # 注意：sel_mask_logits 经过切片后不一定是 contiguous 的，
        # 使用 reshape 而不是 view，避免 stride 不兼容导致的 RuntimeError。
        # 展平为 (B*Q, H, W) 调用 maskscore，得到 (B*Q,) 再还原为 (B,Q)
        mask_scores_flat = maskscore(sel_mask_logits.reshape(Bm * Qm, Hm, Wm), None)  # (B*Q,)
        mask_scores_bq = mask_scores_flat.view(Bm, Qm)                                 # (B, Q)

        # 最终用于筛选的综合分数：分类置信度 * mask 质量分
        final_scores_bq = max_probs_bq * mask_scores_bq
        return final_scores_bq

    def init(
        self,
        queries_bctqT: torch.Tensor,   # (B,C,T,Q)
        masks_bqtThw: torch.Tensor,    # (B,Q,T,H,W)
        logits_btqTc: Optional[torch.Tensor] = None,  # (B,T,Q,num_classes+1)
        init_frame_ids=(0, 1, 2),
       
    ) -> Dict[str, torch.Tensor]:

        """
        Returns a dict:
          - fused_last3: (B,C,3,Q) for frames 3,4,5
          - mem_tokens_final: (B, mem_frames*Q, C) memory tokens at end (optional debug)
        """
        B, C, T, Q = queries_bctqT.shape
        assert Q == self.Q, f"Expected Q={self.Q}, got {Q}"
        assert masks_bqtThw.shape[0] == B and masks_bqtThw.shape[1] == Q and masks_bqtThw.shape[2] == T
        for fid in init_frame_ids:
            assert 0 <= int(fid) < T, f"init_frame_id={fid} out of range for T={T}"

        # reset & init memory from first 3 frames
        self.reset_memory()
        for fid in init_frame_ids:
            q = queries_bctqT[:, :, fid:fid+1, :]          # (B,C,1,Q)
            m = masks_bqtThw[:, :, fid:fid+1, :, :]        # (B,Q,1,H,W)
            tokens = self._encode_frame_tokens(q, m, frame_id=fid)  # (B,Q,C)
            scores_bq = None
            if (self.topk is not None) and (logits_btqTc is not None):
                # logits_btqTc: (B,T,Q,num_classes+1) -> 当前帧 (B,Q,num_classes+1)
                logits_bqc = logits_btqTc[:, fid, :, :]
                scores_bq = self._compute_final_scores(logits_bqc, m)
            # 若未提供 logits 或未设置 topk，则退化为整帧写入（FIFO）
            self._write_memory(tokens, frame_id=fid, scores_bq=scores_bq, topk=self.topk)

    def forward(
        self,
        queries_bctqT: torch.Tensor,
        masks_bqtThw: torch.Tensor,
        query_denoisy: torch.Tensor,
        logits_btqTc: Optional[torch.Tensor] = None,   # (B,T,Q,num_classes+1)
        proc_frame_ids=(3, 4, 5),
    ):  # queries: (B,C,T,Q)
        """Process `proc_frame_ids` sequentially.

        Notes:
          - query_denoisy is expected to be aligned with proc_frame_ids in order.
            i.e. query_denoisy[:,:,i,:] corresponds to proc_frame_ids[i].
        """

        B, C, T, Q = queries_bctqT.shape
        assert Q == self.Q, f"Expected Q={self.Q}, got {Q}"
        assert masks_bqtThw.shape[0] == B and masks_bqtThw.shape[1] == Q and masks_bqtThw.shape[2] == T

        proc_frame_ids = list(proc_frame_ids)
        for fid in proc_frame_ids:
            assert 0 <= int(fid) < T, f"proc_frame_id={fid} out of range for T={T}"
        assert query_denoisy.shape[0] == B and query_denoisy.shape[1] == C and query_denoisy.shape[3] == Q
        assert query_denoisy.shape[2] == len(proc_frame_ids), (
            f"query_denoisy has N={query_denoisy.shape[2]} frames, but proc_frame_ids has len={len(proc_frame_ids)}"
        )

        # process frames sequentially with FIFO updates
        fused_out=[]
        for i, fid in enumerate(proc_frame_ids):
            q = queries_bctqT[:, :, fid:fid+1, :]          # (B,C,1,Q)
            m = masks_bqtThw[:, :, fid:fid+1, :, :]        # (B,Q,1,H,W)
            q_d = query_denoisy[:, :, i:i+1, :]            # (B,C,1,Q)
            cur_tokens = self._encode_frame_tokens(q, m, frame_id=fid)  # (B,Q,C)

            # 写入 memory：若设置了 topk 且提供了 logits，则按综合分数筛选 Top‑K query
            scores_bq = None
            if (self.topk is not None) and (logits_btqTc is not None):
                logits_bqc = logits_btqTc[:, fid, :, :]   # (B,Q,num_classes+1)
                scores_bq = self._compute_final_scores(logits_bqc, m)

            # FIFO write: deque(maxlen=3) auto pops oldest
            self._write_memory(cur_tokens, frame_id=fid, scores_bq=scores_bq, topk=self.topk) #更新query
            frame_ids = torch.tensor([fid], device=q_d.device, dtype=torch.long)  # (1,)
            if q_d.dim() == 4:
                # (B,C,1,Q) -> (B,1,Q,C)
                q_d =q_d.permute(0, 2, 3, 1).contiguous()
            elif q_d.dim() == 3:
                # (B,C,Q) -> (B,1,Q,C)
                q_d= q_d.permute(0, 2, 1).unsqueeze(1).contiguous()
            else:
                raise ValueError("q_d must be (B,C,1,Q) or (B,C,Q)")
            q_d = self.temporal(q_d, frame_ids) #加入位置编码
            q_d=q_d.squeeze(1) # (B,Q,C)
            cur_fused = self._read_memory(q_d)     #用去噪后的query去检索memory bank(B,Q,C)
     
            fused_out.append(cur_fused)    # store output

        # (B,N,Q,C) -> (B,C,N,Q)
        fused_out = torch.stack(fused_out, dim=1)                 # (B,N,Q,C)
        fused_out = fused_out.permute(0, 3, 1, 2).contiguous()    # (B,C,N,Q)
        return fused_out
        # mem_tokens_final = torch.cat([item["tokens"] for item in self._memory_bank], dim=1)  # (B, 3*Q, C)
        # return {
        #     "fused_last3": fused_last3,          # (B,C,3,Q)
        #     "mem_tokens_final": mem_tokens_final # (B,3*Q,C)
        # }
