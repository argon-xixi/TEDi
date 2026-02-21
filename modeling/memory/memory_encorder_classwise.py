import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

"""Class-wise query memory encoder.

设计目标（根据你的需求）：

1. **按类别维护 query 序列的 memory bank**
   - memory 不再按帧/视频划分，而是按语义类别划分；
   - 每个类别维护最多 `per_class_max_queries` 条 query（默认 5 条）；
   - memory 在一个 device 上对所有 batch **共享**，不区分 batch 维度。

2. **更新规则（在模块外部、基于 GT / 匹配结果调用）**
   - 外部先通过 Hungarian 匹配等方式，拿到每个 query 对应的语义类别 label；
   - 调用 `update_memory(tokens_bqc, class_labels_bq)` 进行更新；
   - 对于属于同一类别的新 query：
     - 若该类别 memory 数量 < `per_class_max_queries`，则直接 append；
     - 否则：与该类别已有的所有 memory query 计算相似度（cosine），
       找到**相似度最高**的旧 query，将其替换为新的 query
       （“去掉相似度最高的进行更新”）。

3. **读取规则（cross-attention 结构保持不变）**
   - 读取时，把当前 device 上所有类别的所有 query 按类别拼接：
       `mem_LC = cat([mem_c for all classes], dim=0)`，形状 (L_mem, C)；
   - 对于输入的 query tokens `cur_tokens_bqc`（B, Q, C）：
       - 将 memory broadcast 到 batch 维度，得到 (B, L_mem, C)；
       - 通过若干层 `QueryMemoryLayer` (self-attn + cross-attn + MLP)
         融合，输出同形状 (B, Q, C)。

4. **编码部分保持与原 memory_encorder 一致**
   - 继续复用 `MaskTokenEncoder` + `TemporalEncoding` + `GatedFuser`，
     将 (query, mask) 编成 (B, Q, C) 的 token；
   - 你可以：
       - 用本模块的 `_encode_frame_tokens(...)` 编码当前帧，
         然后：
           - 用 `read_memory(...)` 从 class-wise memory 里读取；
           - 再在外部根据 GT / 预测类别调用 `update_memory(...)`；
       - 或者直接把已经是 (B, Q, C) 的 query 特征送进 `read_memory`/`update_memory`。

5. **训练 vs 推理（与你的设定 1-C 一致）**
   - 训练阶段：建议使用 Hungarian 匹配结果得到的 GT 类别 label，
     通过 `update_memory` 更新；
   - 推理阶段：可以用 `pred_logits.argmax(-1)` 得到 pseudo label，
     然后同样通过 `update_memory` 更新。

本文件只负责 **class-wise memory 的维护与读写逻辑**，
不直接耦合到 MaskFormer / tracker 的 loss 计算中，
方便你在外部灵活接入 GT / 匹配结果。
"""

from .memory_encorder import (
    MaskTokenEncoder,
    TemporalEncoding,
    GatedFuser,
    QueryMemoryLayer,
)


class ClasswiseQueryMemoryModule(nn.Module):
    """按类别共享的 query-level memory 模块。

    核心接口：

    - ``encode_frame_tokens(q_bctq, m_bqthw, frame_id) -> (B, Q, C)``
        复用原有 MaskTokenEncoder + TemporalEncoding + GatedFuser。

    - ``read_memory(cur_tokens_bqc) -> (B, Q, C)``
        将当前 tokens 与 **所有类别的 memory** 做多层 cross-attention 融合。

    - ``update_memory(tokens_bqc, class_labels_bq, ignore_label=None)``
        根据类别标签对 **device 级别共享的 class-wise memory bank** 进行更新，
        每类最多保留 ``per_class_max_queries`` 条 query，
        当达到上限时，用“与新 query 最相似的旧 query”进行替换更新。

    使用建议：

    1）训练阶段（你的设定 1-C）：
        - 在计算 tracker / segmentation loss 之后，
          利用 Hungarian 匹配结果拿到每个 query 的 GT 类别 label；
        - 把对应的 query 特征整理成 (B, Q, C)，label 整理成 (B, Q)，
          调用 ``update_memory(tokens_bqc, class_labels_bq)``。

    2）推理阶段：
        - 可以用预测 ``pred_logits.argmax(-1)`` 当作 pseudo label，
          然后同样调用 ``update_memory``；

    3）注意：
        - 本模块内部 **不区分 batch 维度**，memory 是按 device 共享的；
        - 可以通过 ``reset_memory()`` 在需要的时候清空 memory。
    """

    def __init__(
        self,
        C: int = 256,
        Q: int = 10,
        num_classes: int = 8,
        per_class_max_queries: int = 10,
        fuser_hidden: int = 1024,
        down_h: int = 32,
        down_w: int = 32,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_frames: int = 1024,
        detach_memory: bool = True,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        self.C = C
        self.Q = Q
        self.num_classes = num_classes
        self.per_class_max_queries = per_class_max_queries
        self.detach_memory = detach_memory

        # 用于多样性更新策略的时间戳（LRU 替换）
        # 仅在新分支 update_memory_diverse 中使用，原有 update_memory 行为保持不变
        self._global_step: int = 0

        # 编码部分（与原 memory_encorder 保持一致）
        self.mask_encoder = MaskTokenEncoder(C=C, down_h=down_h, down_w=down_w)
        self.temporal = TemporalEncoding(C=C, max_frames=max_frames)
        self.fuser = GatedFuser(C=C, hidden=fuser_hidden, drop=dropout)

        # 多层 query-level memory attention（结构与原版一致）
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            QueryMemoryLayer(
                C=C,
                num_heads=num_heads,
                dim_feedforward=fuser_hidden,
                dropout=dropout,
            )
            for _ in range(self.num_layers)
        ])

        # 每个 device 维护一个 dict: class_id -> Tensor(K_c, C)
        self._memory_bank: Dict[torch.device, Dict[int, torch.Tensor]] = {}

        # 对应每个 memory slot 的“上次更新 step”，用于 LRU 替换策略
        # 结构: device -> { class_id -> Tensor(K_c,) }
        self._memory_last_update: Dict[torch.device, Dict[int, torch.Tensor]] = {}

    # ------------------------------------------------------------------
    # 内部工具函数：bank 管理
    # ------------------------------------------------------------------
    def _get_bank(self, device: torch.device) -> Dict[int, torch.Tensor]:
        if device not in self._memory_bank:
            self._memory_bank[device] = {}
        return self._memory_bank[device]

    def _get_last_update_bank(self, device: torch.device) -> Dict[int, torch.Tensor]:
        """获取 / 初始化当前 device 上的 last_update 字典。"""

        if device not in self._memory_last_update:
            self._memory_last_update[device] = {}
        return self._memory_last_update[device]

    def reset_memory(self) -> None:
        """清空所有 device 上的 memory bank。"""

        self._memory_bank.clear()
        self._memory_last_update.clear()
        self._global_step = 0

    # ------------------------------------------------------------------
    # 编码：与原 memory_encorder 中的 _encode_frame_tokens 一致
    # ------------------------------------------------------------------
    def encode_frame_tokens(
        self,
        q_bctq: torch.Tensor,  # (B,C,1,Q) 或 (B,C,Q)
        m_bqthw: torch.Tensor,  # (B,Q,1,H,W)
        frame_id: int,
    ) -> torch.Tensor:
        """将单帧的 query+mask 编成 (B, Q, C) 的 token。

        - 继续使用 MaskTokenEncoder + TemporalEncoding + GatedFuser；
        - 与原 ``VideoQueryMemoryModule._encode_frame_tokens`` 兼容。
        """

        # # query -> (B,1,Q,C)
        if q_bctq.dim() == 4:
            # (B,C,1,Q) -> (B,1,Q,C)
            q_btqc = q_bctq.permute(0, 2, 3, 1).contiguous()
        elif q_bctq.dim() == 3:
            # (B,C,Q) -> (B,1,Q,C)
            q_btqc = q_bctq.permute(0, 2, 1).unsqueeze(1).contiguous()
        else:
            raise ValueError("q_bctq must be (B,C,1,Q) or (B,C,Q)")

        # # mask -> (B,1,Q,C)
        m_btqc = self.mask_encoder(m_bqthw)  # (B,1,Q,C)

        # # temporal encoding (absolute frame id)
        # frame_ids = torch.tensor([frame_id], device=q_bctq.device, dtype=torch.long)  # (1,)
        # q_btqc = self.temporal(q_btqc, frame_ids)
        # m_btqc = self.temporal(m_btqc, frame_ids)

        # gated fusion -> (B,1,Q,C)
        fused_btqc = self.fuser(q_btqc, m_btqc)

        # return (B,Q,C)
        return fused_btqc.squeeze(1)

    # ------------------------------------------------------------------
    # 读取：当前 tokens 对所有类别的 memory 做 cross-attention
    # ------------------------------------------------------------------
    def _collect_all_memory_tokens(self, device: torch.device) -> Optional[torch.Tensor]:
        """将当前 device 上所有类别的 memory 按类别 id 排序后拼接。

        返回：
            - mem_LC: (L_mem, C) 或
            - None: 若当前没有任何 memory。
        """

        bank = self._get_bank(device)
        if not bank:
            return None

        # 按 class_id 排序，方便 debug/复现
        mem_list = []
        for cls_id in sorted(bank.keys()):
            mem_cls = bank[cls_id]
            if mem_cls.numel() == 0:
                continue
            # mem_cls: (K_c, C)
            mem_list.append(mem_cls)

        if len(mem_list) == 0:
            return None

        mem_LC = torch.cat(mem_list, dim=0)  # (L_mem, C)
        return mem_LC

    def read_memory(self, cur_tokens_bqc: torch.Tensor) -> torch.Tensor:
        """用当前 tokens (B,Q,C) 从 class-wise memory bank 中读取信息。

        - 若当前 device 上没有 memory，则直接返回输入；
        - 否则：
            1) 收集所有类别的 memory，得到 mem_LC (L_mem, C)；
            2) broadcast 到 batch 维度，得到 mem_bLc (B,L_mem,C)；
            3) 通过若干层 QueryMemoryLayer 进行 self-attn+cross-attn+MLP 融合。
        """

        dev = cur_tokens_bqc.device
        mem_LC = self._collect_all_memory_tokens(dev)
        if mem_LC is None:
            return cur_tokens_bqc

        B, _, _ = cur_tokens_bqc.shape
        mem_bLc = mem_LC.unsqueeze(0).expand(B, -1, -1).contiguous()  # (B,L_mem,C)

        x = cur_tokens_bqc
        for layer in self.layers:
            x = layer(x, mem_bLc)
        return x

    # ------------------------------------------------------------------
    # 更新：在模块外部根据 GT / 预测标签调用
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update_memory(
        self,
        tokens_bqc: torch.Tensor,     # (B,Q,C)
        class_labels_bq: torch.Tensor,  # (B,Q)，语义类别 id
        ignore_label: Optional[int] = None,
    ) -> None:
        """基于类别标签更新 class-wise memory bank（device 级别共享）。

        参数：
            tokens_bqc: 当前 batch 的 query 特征，形状 (B,Q,C)；
            class_labels_bq: 对应的类别 id，形状 (B,Q)，值域通常在 [0, num_classes-1]；
            ignore_label: 若不为 None，则忽略等于该值的样本
                          （例如可以用来跳过背景/未标注类别）。

        规则：
            - 对每个 (b,q) 形成一条样本 (token, class_id)；
            - 按类聚合后逐条更新：
                * 若该类 memory 数量 < per_class_max_queries：append；
                * 否则：
                    1) 与该类所有已有 memory 计算 cosine 相似度；
                    2) 找到相似度最高的旧条目，用新 token 替换它。
        """

        assert tokens_bqc.shape[:2] == class_labels_bq.shape[:2], (
            f"tokens_bqc shape {tokens_bqc.shape} and class_labels_bq shape {class_labels_bq.shape} mismatch"
        )

        dev = tokens_bqc.device
        bank = self._get_bank(dev)

        if self.detach_memory:
            tokens_bqc = tokens_bqc.detach()

        B, Q, C = tokens_bqc.shape

        # 展平 batch 维度，得到 (N, C) 与 (N,)
        tokens_NC = tokens_bqc.view(B * Q, C)
        labels_N = class_labels_bq.view(B * Q)

        # 有效样本：label >= 0，并且（若指定 ignore_label）label != ignore_label
        valid_mask = labels_N >= 0
        if ignore_label is not None:
            valid_mask &= labels_N != ignore_label

        valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            return

        labels_valid = labels_N[valid_indices]
        tokens_valid = tokens_NC[valid_indices]  # (N_valid, C)

        # 按类别逐类更新
        for cls_id in labels_valid.unique().tolist():
            cls_id_int = int(cls_id)

            # 只保留 [0, num_classes-1] 范围内的类别（防止异常 label）
            if cls_id_int < 0 or cls_id_int >= self.num_classes:
                continue

            cls_mask = labels_valid == cls_id
            if not torch.any(cls_mask):
                continue

            new_tokens_cls = tokens_valid[cls_mask]  # (M_c, C)

            if cls_id_int not in bank:
                # 该类还没有任何 memory，直接初始化，但需要遵守
                # per_class_max_queries 的上限约束。
                # 注意：不保留 batch 维度，直接作为 (K_c, C)
                if new_tokens_cls.shape[0] > self.per_class_max_queries:
                    bank[cls_id_int] = new_tokens_cls[: self.per_class_max_queries].clone()
                else:
                    bank[cls_id_int] = new_tokens_cls.clone()
                continue

            # 已有 memory：mem_old (K_c, C)
            mem_old = bank[cls_id_int]

            # 逐条样本更新（也可以改成更复杂的聚合策略，这里保持简单直观）
            for tok in new_tokens_cls:
                tok = tok.unsqueeze(0)  # (1, C)

                if mem_old.numel() == 0:
                    mem_old = tok.clone()
                elif mem_old.shape[0] < self.per_class_max_queries:
                    # 未满：直接 append
                    mem_old = torch.cat([mem_old, tok], dim=0)  # (K_c+1, C)
                else:
                    # 已满：与所有 memory 计算 cosine 相似度，替换最相似的条目
                    # mem_old: (K_c, C), tok: (1, C)
                    # F.cosine_similarity 会在 dim=-1 上计算，相当于对每一行算 cos sim
                    sims = F.cosine_similarity(mem_old, tok.expand_as(mem_old), dim=-1)  # (K_c,)
                    max_idx = torch.argmax(sims).item()
                    mem_old[max_idx] = tok.squeeze(0)

            bank[cls_id_int] = mem_old

    # ------------------------------------------------------------------
    # 新分支：带“多样性约束”的更新策略（保留原有 update_memory 不变）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update_memory_diverse(
        self,
        tokens_bqc: torch.Tensor,       # (B,Q,C)
        class_labels_bq: torch.Tensor,  # (B,Q)
        ignore_label: Optional[int] = None,
        sim_threshold: float = 0.8,
        momentum: float = 0.1,
    ) -> None:
        """基于类别标签 + 多样性约束 更新 class-wise memory bank。

        相比原始的 ``update_memory``：
        - 保留“每类最多 per_class_max_queries 条 prototype”的设计；
        - 在 memory 未满时，仍然是**直接追加**高置信度样本；
        - 在 memory 已满时：
            * 若新 token 与某一 prototype 的余弦相似度 >= sim_threshold：
                - 视为同一模式，对该 prototype 做动量更新；
            * 否则（与所有 prototype 都比较远）：
                - 认为是新模式，替换掉该类中“最久未更新”的 prototype（LRU）。
        """

        assert tokens_bqc.shape[:2] == class_labels_bq.shape[:2], (
            f"tokens_bqc shape {tokens_bqc.shape} and class_labels_bq shape {class_labels_bq.shape} mismatch"
        )

        dev = tokens_bqc.device
        bank = self._get_bank(dev)
        last_update_bank = self._get_last_update_bank(dev)

        if self.detach_memory:
            tokens_bqc = tokens_bqc.detach()

        B, Q, C = tokens_bqc.shape

        tokens_NC = tokens_bqc.view(B * Q, C)
        labels_N = class_labels_bq.view(B * Q)

        # 有效样本：label >= 0，并且（若指定 ignore_label）label != ignore_label
        valid_mask = labels_N >= 0
        if ignore_label is not None:
            valid_mask &= labels_N != ignore_label

        valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            return

        labels_valid = labels_N[valid_indices]
        tokens_valid = tokens_NC[valid_indices]  # (N_valid, C)

        # 按类别逐类更新
        for cls_id in labels_valid.unique().tolist():
            cls_id_int = int(cls_id)

            # 只保留 [0, num_classes-1] 范围内的类别（防止异常 label）
            if cls_id_int < 0 or cls_id_int >= self.num_classes:
                continue

            cls_mask = labels_valid == cls_id
            if not torch.any(cls_mask):
                continue

            new_tokens_cls = tokens_valid[cls_mask]  # (M_c, C)

            # 该类还没有任何 memory：直接初始化为这些 token（截断到上限）
            if cls_id_int not in bank:
                if new_tokens_cls.shape[0] > self.per_class_max_queries:
                    mem_init = new_tokens_cls[: self.per_class_max_queries].clone()
                else:
                    mem_init = new_tokens_cls.clone()

                # 归一化存储，便于使用 cosine 相似度
                mem_init = F.normalize(mem_init, dim=-1)
                bank[cls_id_int] = mem_init

                # 初始化对应的 last_update 时间戳
                K_c = mem_init.shape[0]
                last_update_bank[cls_id_int] = torch.full(
                    (K_c,),
                    fill_value=float(self._global_step),
                    device=dev,
                    dtype=torch.float32,
                )
                continue

            # 已有 memory：mem_old (K_c, C)，以及对应的 last_update (K_c,)
            mem_old = bank[cls_id_int]
            last_update = last_update_bank.get(cls_id_int, None)
            if last_update is None or last_update.shape[0] != mem_old.shape[0]:
                # 若形状不一致，重新初始化时间戳
                last_update = torch.full(
                    (mem_old.shape[0],),
                    fill_value=float(self._global_step),
                    device=dev,
                    dtype=torch.float32,
                )

            for tok in new_tokens_cls:
                # 每处理一个 token，增加全局 step（用于 LRU）
                self._global_step += 1

                tok = tok.unsqueeze(0)  # (1, C)
                tok_norm = F.normalize(tok, dim=-1)  # (1, C)

                if mem_old.numel() == 0:
                    mem_old = tok_norm.clone()
                    last_update = torch.tensor(
                        [float(self._global_step)],
                        device=dev,
                        dtype=torch.float32,
                    )
                elif mem_old.shape[0] < self.per_class_max_queries:
                    # 未满：直接 append
                    mem_old = torch.cat([mem_old, tok_norm], dim=0)  # (K_c+1, C)
                    last_update = torch.cat(
                        [
                            last_update,
                            torch.tensor(
                                [float(self._global_step)],
                                device=dev,
                                dtype=torch.float32,
                            ),
                        ],
                        dim=0,
                    )
                else:
                    # 已满：根据相似度 & LRU 决定是更新最近邻还是替换旧 prototype
                    # mem_old: (K_c, C) 已假定归一化
                    mem_norm = F.normalize(mem_old, dim=-1)
                    sims = F.cosine_similarity(mem_norm, tok_norm.expand_as(mem_norm), dim=-1)  # (K_c,)
                    max_sim, max_idx = torch.max(sims, dim=0)
                    max_sim_val = float(max_sim.item())
                    max_idx_val = int(max_idx.item())

                    if max_sim_val >= sim_threshold:
                        # 与最近邻 prototype 很接近：做动量更新，保持“主模式”代表性
                        mem_old[max_idx_val] = F.normalize(
                            (1.0 - momentum) * mem_old[max_idx_val] + momentum * tok_norm.squeeze(0),
                            dim=-1,
                        )
                        last_update[max_idx_val] = float(self._global_step)
                    else:
                        # 与所有 prototype 都比较远：认为是新模式，替换最久未更新的那个（LRU）
                        lru_idx = int(torch.argmin(last_update).item())
                        mem_old[lru_idx] = tok_norm.squeeze(0)
                        last_update[lru_idx] = float(self._global_step)

            bank[cls_id_int] = mem_old
            last_update_bank[cls_id_int] = last_update

    # ------------------------------------------------------------------
    # 一个简单的 forward 封装（可选）：encode + read
    # ------------------------------------------------------------------
    def forward(
        self,
        q_bctq: torch.Tensor,      # (B,C,1,Q) 或 (B,C,Q)
        m_bqthw: torch.Tensor,      # (B,Q,1,H,W)
        frame_id: int,
        *,
        return_tokens: bool = False,
    ):
        """便捷接口：单帧的 encode + read。

        - 先通过 ``encode_frame_tokens`` 得到 (B,Q,C)；
        - 再通过 ``read_memory`` 从全局 class-wise memory 中读取信息；
        - 不在内部做 memory 更新
          你可以在外部拿到 encode 后的 tokens再结合 GT 调用 ``update_memory``。

        返回：
            - 若 ``return_tokens=False``：返回 fused_tokens_bqc (B,Q,C)
            - 若 ``return_tokens=True``：返回 (fused_tokens_bqc, encoded_tokens_bqc)
        """

        encoded_bqc = self.encode_frame_tokens(q_bctq, m_bqthw, frame_id)
        fused_bqc = self.read_memory(encoded_bqc)

        if return_tokens:
            return fused_bqc, encoded_bqc
        return fused_bqc


__all__ = [
    "ClasswiseQueryMemoryModule",
]
