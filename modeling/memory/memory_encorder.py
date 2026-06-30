import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Optional, Dict

def maskscore(sel_mask_logits: torch.Tensor, pred_masks_bin: Optional[torch.Tensor] = None, mask_bin_thresh: float = 0.0) -> torch.Tensor:

    pred_masks_bin = (sel_mask_logits > mask_bin_thresh).float()
    mask_prob = sel_mask_logits.sigmoid()
    numer = (mask_prob.flatten(1) * pred_masks_bin.flatten(1)).sum(1)
    denom = pred_masks_bin.flatten(1).sum(1).clamp_min(1e-6)
    mask_scores = numer / denom
    return mask_scores

class MaskTokenEncoder(nn.Module):

    def __init__(self, C: int, down_h: int = 32, down_w: int = 32):
        super().__init__()
        self.down_h = down_h
        self.down_w = down_w
        self.proj = nn.Linear(down_h * down_w, C)

    def forward(self, masks_bqthw: torch.Tensor) -> torch.Tensor:
        B, Q, T, H, W = masks_bqthw.shape

        x = masks_bqthw.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * T * Q, 1, H, W)

        x = torch.tanh(x)

        x = F.interpolate(
            x,
            size=(self.down_h, self.down_w),
            mode="bilinear",
            align_corners=False,
        )
        x = x.flatten(1)
        x = self.proj(x)
        x = x.view(B, T, Q, -1)
        return x

class TemporalEncoding(nn.Module):

    def __init__(self, C: int, max_frames: int = 1024):
        super().__init__()
        self.time_emb = nn.Embedding(max_frames, C)

    def forward(self, x_btqc: torch.Tensor, frame_ids_t: torch.Tensor) -> torch.Tensor:

        assert frame_ids_t.dim() == 1
        T = x_btqc.shape[1]
        assert frame_ids_t.numel() == T

        e = self.time_emb(frame_ids_t).view(1, T, 1, -1)
        return x_btqc + e

class GatedFuser(nn.Module):

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
        z = torch.cat([q_btqc, m_btqc], dim=-1)
        z = self.norm(z)
        g = self.gate(z)
        d = self.mlp(z)
        return q_btqc + g * d

class QueryMemoryLayer(nn.Module):

    def __init__(self, C: int, num_heads: int = 8, dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=C,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=C,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

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
        x = self.norm1(q_bqc)
        x, _ = self.self_attn(x, x, x, need_weights=False)
        q_bqc = q_bqc + self.dropout1(x)

        x = self.norm2(q_bqc)
        x, _ = self.cross_attn(x, mem_bLc, mem_bLc, need_weights=False)
        q_bqc = q_bqc + self.dropout2(x)

        x = self.norm3(q_bqc)
        x = self.linear2(self.dropout3(self.activation(self.linear1(x))))
        q_bqc = q_bqc + x
        return q_bqc

class VideoQueryMemoryModule(nn.Module):

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

        self.topk = topk

        self.mask_encoder = MaskTokenEncoder(C=C, down_h=down_h, down_w=down_w)
        self.temporal = TemporalEncoding(C=C, max_frames=max_frames)
        self.fuser = GatedFuser(C=C, hidden=fuser_hidden, drop=dropout)

        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            QueryMemoryLayer(C=C, num_heads=num_heads, dim_feedforward=fuser_hidden, dropout=dropout)
            for _ in range(self.num_layers)
        ])

        self._memory_bank = {}

    def _get_bank(self, device):
        if device not in self._memory_bank:
            self._memory_bank[device] = deque(maxlen=self.mem_frames)
        return self._memory_bank[device]

    def reset_memory(self):
        self._memory_bank.clear()

    def _encode_frame_tokens(
        self,
        q_bctq: torch.Tensor,
        m_bqthw: torch.Tensor,
        frame_id: int,
    ) -> torch.Tensor:

        if q_bctq.dim() == 4:

            q_btqc = q_bctq.permute(0, 2, 3, 1).contiguous()
        elif q_bctq.dim() == 3:

            q_btqc = q_bctq.permute(0, 2, 1).unsqueeze(1).contiguous()
        else:
            raise ValueError("q_bctq must be (B,C,1,Q) or (B,C,Q)")

        m_btqc = self.mask_encoder(m_bqthw)

        frame_ids = torch.tensor([frame_id], device=q_bctq.device, dtype=torch.long)
        q_btqc = self.temporal(q_btqc, frame_ids)
        m_btqc = self.temporal(m_btqc, frame_ids)

        fused_btqc = self.fuser(q_btqc, m_btqc)

        return fused_btqc.squeeze(1)

    def _read_memory(self, cur_tokens_bqc: torch.Tensor) -> torch.Tensor:
        dev = cur_tokens_bqc.device
        bank = self._get_bank(dev)
        if len(bank) == 0:
            return cur_tokens_bqc
        # Attend to all query tokens retained by the temporal memory bank.
        mem_tokens = torch.cat([item["tokens"] for item in bank], dim=1)
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

        if (scores_bq is not None) and (topk is not None):
            B, Q, C = tokens_bqc.shape
            assert scores_bq.shape[:2] == (B, Q), f"scores_bq shape {scores_bq.shape} mismatch with tokens {tokens_bqc.shape}"
            k = min(topk, Q)

            topk_scores, topk_idx = torch.topk(scores_bq, k=k, dim=1)
            batch_idx = torch.arange(B, device=dev).unsqueeze(1).expand(-1, k)

            tokens_topk = tokens_bqc[batch_idx, topk_idx, :]
            bank.append({"tokens": tokens_topk, "frame_id": frame_id, "scores": topk_scores})
        else:

            bank.append({"tokens": tokens_bqc, "frame_id": frame_id})

    def _compute_final_scores(
        self,
        logits_bqc: torch.Tensor,
        m_bq1hw: torch.Tensor,
    ) -> torch.Tensor:

        probs_btqc = torch.softmax(logits_bqc, dim=-1)[:, :, :-1]
        max_probs_bq, _ = probs_btqc.max(dim=-1)

        sel_mask_logits = m_bq1hw[:, :, 0, :, :]
        Bm, Qm, Hm, Wm = sel_mask_logits.shape

        mask_scores_flat = maskscore(sel_mask_logits.reshape(Bm * Qm, Hm, Wm), None)
        mask_scores_bq = mask_scores_flat.view(Bm, Qm)

        final_scores_bq = max_probs_bq * mask_scores_bq
        return final_scores_bq

    def init(
        self,
        queries_bctqT: torch.Tensor,
        masks_bqtThw: torch.Tensor,
        logits_btqTc: Optional[torch.Tensor] = None,
        init_frame_ids=(0, 1, 2),

    ) -> Dict[str, torch.Tensor]:

        B, C, T, Q = queries_bctqT.shape
        assert Q == self.Q, f"Expected Q={self.Q}, got {Q}"
        assert masks_bqtThw.shape[0] == B and masks_bqtThw.shape[1] == Q and masks_bqtThw.shape[2] == T
        for fid in init_frame_ids:
            assert 0 <= int(fid) < T, f"init_frame_id={fid} out of range for T={T}"

        self.reset_memory()
        for fid in init_frame_ids:
            q = queries_bctqT[:, :, fid:fid+1, :]
            m = masks_bqtThw[:, :, fid:fid+1, :, :]
            tokens = self._encode_frame_tokens(q, m, frame_id=fid)
            scores_bq = None
            if (self.topk is not None) and (logits_btqTc is not None):

                logits_bqc = logits_btqTc[:, fid, :, :]
                scores_bq = self._compute_final_scores(logits_bqc, m)

            self._write_memory(tokens, frame_id=fid, scores_bq=scores_bq, topk=self.topk)

    def forward(
        self,
        queries_bctqT: torch.Tensor,
        masks_bqtThw: torch.Tensor,
        query_denoisy: torch.Tensor,
        logits_btqTc: Optional[torch.Tensor] = None,
        proc_frame_ids=(3, 4, 5),
    ):

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

        fused_out=[]
        for i, fid in enumerate(proc_frame_ids):
            q = queries_bctqT[:, :, fid:fid+1, :]
            m = masks_bqtThw[:, :, fid:fid+1, :, :]
            q_d = query_denoisy[:, :, i:i+1, :]
            cur_tokens = self._encode_frame_tokens(q, m, frame_id=fid)

            scores_bq = None
            if (self.topk is not None) and (logits_btqTc is not None):
                logits_bqc = logits_btqTc[:, fid, :, :]
                scores_bq = self._compute_final_scores(logits_bqc, m)

            self._write_memory(cur_tokens, frame_id=fid, scores_bq=scores_bq, topk=self.topk)
            frame_ids = torch.tensor([fid], device=q_d.device, dtype=torch.long)
            if q_d.dim() == 4:

                q_d =q_d.permute(0, 2, 3, 1).contiguous()
            elif q_d.dim() == 3:

                q_d= q_d.permute(0, 2, 1).unsqueeze(1).contiguous()
            else:
                raise ValueError("q_d must be (B,C,1,Q) or (B,C,Q)")
            q_d = self.temporal(q_d, frame_ids)
            q_d=q_d.squeeze(1)
            cur_fused = self._read_memory(q_d)

            fused_out.append(cur_fused)

        fused_out = torch.stack(fused_out, dim=1)
        fused_out = fused_out.permute(0, 3, 1, 2).contiguous()
        return fused_out

