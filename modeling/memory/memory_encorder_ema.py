import torch
import torch.nn as nn
from typing import Dict, List, Tuple

# 直接复用原文件里的基础模块，避免代码重复
from .memory_encorder import (
    MaskTokenEncoder,
    TemporalEncoding,
    GatedFuser,
    QueryMemoryLayer,
)


class VideoQueryMemoryModuleEMA(nn.Module):
    """基于 EMA 的 query-level memory 模块。

    和原来的 ``VideoQueryMemoryModule`` 区别：

    - 原版：memory bank 是一个长度为 ``mem_frames`` 的 FIFO 队列
      （这里 mem_frames=3），处理 frame 3,4,5 时，不断 ``append`` 新帧，
      自动弹出最早帧。
    - 本版：memory bank **固定就是 3 槽**，不再 FIFO。
      - 初始用前三帧 (frame 0,1,2) 初始化 3 个槽位；
      - 之后每来一帧新图（frame 3,4,5），
        用 **EMA** 去更新对应槽位：

          ``mem[i] = (1-m)*mem[i] + m*cur_tokens``

        这样始终只有 3 份“记忆”，但这 3 份是之前所有对应帧的指数滑动平均。

    默认假设：
      - 输入依然是 6 帧：前 3 帧初始化 memory，后 3 帧做 tracking。
      - 接口保持和原模块一致，方便在 ``MaskFormerModel_track_memorybank`` 里直接替换。
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
        ema_momentum: float = 0.1,
    ):
        super().__init__()
        self.C = C
        self.Q = Q
        self.mem_frames = mem_frames
        self.detach_memory = detach_memory
        self.ema_momentum = ema_momentum

        self.mask_encoder = MaskTokenEncoder(C=C, down_h=down_h, down_w=down_w)
        self.temporal = TemporalEncoding(C=C, max_frames=max_frames)
        self.fuser = GatedFuser(C=C, hidden=fuser_hidden, drop=dropout)

        # 多层 QueryMemoryLayer：每层 = self-attn + cross-attn + MLP
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            QueryMemoryLayer(C=C, num_heads=num_heads, dim_feedforward=fuser_hidden, dropout=dropout)
            for _ in range(self.num_layers)
        ])

        # 每个 device 维护一个长度固定为 mem_frames 的 list
        # list 的每个元素: {"tokens": (B,Q,C), "frame_id": int}
        self._memory_bank: Dict[torch.device, List[Dict[str, torch.Tensor]]] = {}

    # ------------------------------------------------------------------
    # 基础工具函数
    # ------------------------------------------------------------------
    def _get_bank(self, device: torch.device) -> List[Dict[str, torch.Tensor]]:
        if device not in self._memory_bank:
            self._memory_bank[device] = []
        return self._memory_bank[device]

    def reset_memory(self):
        self._memory_bank.clear()

    def _encode_frame_tokens(
        self,
        q_bctq: torch.Tensor,  # (B,C,1,Q) or (B,C,Q)
        m_bqthw: torch.Tensor,  # (B,Q,1,H,W)
        frame_id: int,
    ) -> torch.Tensor:
        """和原版保持一致：query+mask -> temporal -> gated fuse -> (B,Q,C)"""

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
        """当前 tokens (B,Q,C) 对 memory (B,mem_len,C) 做多层 memory attention。"""
        dev = cur_tokens_bqc.device
        bank = self._get_bank(dev)
        if len(bank) == 0:
            return cur_tokens_bqc

        mem_tokens = torch.cat([item["tokens"] for item in bank], dim=1)  # (B,mem_frames*Q,C)
        x = cur_tokens_bqc
        for layer in self.layers:
            x = layer(x, mem_tokens)
        return x

    def _update_memory_slot_ema(self, tokens_bqc: torch.Tensor, frame_id: int, slot_idx: int):
        """用 EMA 更新指定槽位的 memory。

        slot_idx: [0, mem_frames-1]
        """
        if self.detach_memory:
            tokens_bqc = tokens_bqc.detach()

        dev = tokens_bqc.device
        bank = self._get_bank(dev)

        # 理论上在 forward 前已经通过 init() 填满了 mem_frames 个槽，如果不足则补齐
        while len(bank) < self.mem_frames:
            # 用当前 tokens 初始化剩余槽位
            bank.append({"tokens": tokens_bqc.clone(), "frame_id": frame_id})

        assert 0 <= slot_idx < self.mem_frames, f"slot_idx={slot_idx} out of range"

        old_tokens = bank[slot_idx]["tokens"]
        if self.detach_memory:
            old_tokens = old_tokens.detach()

        m = self.ema_momentum
        new_tokens = (1.0 - m) * old_tokens + m * tokens_bqc
        bank[slot_idx]["tokens"] = new_tokens
        bank[slot_idx]["frame_id"] = frame_id

    # ------------------------------------------------------------------
    # 对外接口：init + forward
    # ------------------------------------------------------------------
    def init(
        self,
        queries_bctq6: torch.Tensor,   # (B,C,6,Q)
        masks_bqt6hw: torch.Tensor,    # (B,Q,6,H,W)
        init_frame_ids: Tuple[int, int, int] = (0, 1, 2),
        proc_frame_ids: Tuple[int, int, int] = (3, 4, 5),  # 仅占位，forward 里实际用
    ) -> None:
        """用前三帧初始化固定的 3 个 memory 槽位。

        和原版不同点：
          - 原版：这里把前三帧按 FIFO 写入 deque；
          - 本版：这里直接构造长度为 3 的 list，后续只做 EMA 更新，不再 append/pop。
        """
        B, C, T6, Q = queries_bctq6.shape
        assert T6 == 6, f"Expected 6 frames, got {T6}"
        assert Q == self.Q, f"Expected Q={self.Q}, got {Q}"
        assert masks_bqt6hw.shape[0] == B and masks_bqt6hw.shape[1] == Q and masks_bqt6hw.shape[2] == 6

        self.reset_memory()
        dev = queries_bctq6.device
        bank = self._get_bank(dev)

        for fid in init_frame_ids:
            q = queries_bctq6[:, :, fid:fid + 1, :]       # (B,C,1,Q)
            m = masks_bqt6hw[:, :, fid:fid + 1, :, :]     # (B,Q,1,H,W)
            tokens = self._encode_frame_tokens(q, m, frame_id=fid)  # (B,Q,C)
            if self.detach_memory:
                tokens = tokens.detach()
            bank.append({"tokens": tokens, "frame_id": fid})

        # 保证 bank 长度刚好是 mem_frames
        if len(bank) > self.mem_frames:
            del bank[:-self.mem_frames]

    def forward(
        self,
        queries_bctq6: torch.Tensor,     # (B,C,6,Q)
        masks_bqt6hw: torch.Tensor,      # (B,Q,6,H,W)
        query_denoisy: torch.Tensor,     # (B,C,3,Q) 对应 frame 3,4,5 去噪后的 query
        proc_frame_ids: Tuple[int, int, int] = (3, 4, 5),
    ) -> torch.Tensor:
        """处理后 3 帧，使用 EMA memory：

        - 假定在调用 forward 前，已经调用过一次 ``init``，
          用 frame 0,1,2 初始化了 3 个 memory 槽。
        - 对于 frame 3,4,5：
          - 先编码当前帧 tokens；
          - 用 EMA 更新对应槽位；
          - 再用 ``query_denoisy``（带时间编码）去读 memory，得到融合后的特征。
        """

        fused_out = []

        # 确保已经初始化好 memory
        dev = queries_bctq6.device
        bank = self._get_bank(dev)
        assert len(bank) > 0, "Memory bank is empty. Call `init` before `forward`."

        for i, fid in enumerate(proc_frame_ids):
            # 当前帧的原始 query / mask
            q = queries_bctq6[:, :, fid:fid + 1, :]       # (B,C,1,Q)
            m = masks_bqt6hw[:, :, fid:fid + 1, :, :]     # (B,Q,1,H,W)

            # 对应的去噪后 query（track module 那边传进来的）
            q_d = query_denoisy[:, :, i:i + 1, :]         # (B,C,1,Q)

            # 编码当前帧 tokens
            cur_tokens = self._encode_frame_tokens(q, m, frame_id=fid)  # (B,Q,C)

            # EMA 更新：这里采用简单的一一对应槽位策略：
            #   frame 3 -> slot 0, frame 4 -> slot 1, frame 5 -> slot 2
            slot_idx = i % self.mem_frames
            self._update_memory_slot_ema(cur_tokens, frame_id=fid, slot_idx=slot_idx)

            # 用去噪后的 query 去读 EMA 后的 memory
            frame_ids = torch.tensor([fid], device=q_d.device, dtype=torch.long)  # (1,)
            if q_d.dim() == 4:
                # (B,C,1,Q) -> (B,1,Q,C)
                q_d_btqc = q_d.permute(0, 2, 3, 1).contiguous()
            elif q_d.dim() == 3:
                # (B,C,Q) -> (B,1,Q,C)
                q_d_btqc = q_d.permute(0, 2, 1).unsqueeze(1).contiguous()
            else:
                raise ValueError("q_d must be (B,C,1,Q) or (B,C,Q)")

            q_d_btqc = self.temporal(q_d_btqc, frame_ids)  # 加时间编码
            q_d_bqc = q_d_btqc.squeeze(1)                   # (B,Q,C)

            cur_fused = self._read_memory(q_d_bqc)          # (B,Q,C)
            fused_out.append(cur_fused)

        # stack outputs for frames 3,4,5 -> (B,3,Q,C) -> (B,C,3,Q)
        fused_last3 = torch.stack(fused_out, dim=1)          # (B,3,Q,C)
        fused_last3 = fused_last3.permute(0, 3, 1, 2).contiguous()  # (B,C,3,Q)
        return fused_last3
