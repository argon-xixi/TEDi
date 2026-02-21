import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

"""Class-wise query memory encoder（FIFO 版本）。

本文件是在原 `memory_encorder_classwise.py` 的基础上，
根据你的新需求改写后的 **按 batch 独立、按类别 FIFO 更新** 的版本。

核心设计差异：

1. **不跨 batch 共享 memory（方案 A）**
   - 以当前 mini-batch 内的样本为单位维护 memory：
     对第 b 个样本（通常对应一个视频）维护一份独立的 class-wise memory：
       `memory[b][class_id] -> [该样本内该类的若干 query]`；
   - 同一个 forward 里的不同样本互不共享 memory；
   - 若你希望不同 mini-batch / 不同视频之间也不共享 memory，
     只需要在每个新 batch 开始前调用一次 ``reset_memory()`` 即可。

2. **整体采用 FIFO 更新策略（先进先出）**
   - 每个 (batch_index, class_id) 维护一个长度上限为
     ``per_class_max_queries`` 的队列；
   - 调用 ``update_memory(tokens_bqc, class_labels_bq)`` 时，
     会按 (b, q) 的顺序依次把新 token 追加到对应类别的队列尾部；
   - 如果该类别队列长度超过 ``per_class_max_queries``，
     会自动丢弃最早进入的 token（队列头部），实现 FIFO；
   - 不再使用“找最相似的旧 query 再替换”的策略。

3. **按类别进行更新（外部已完成置信度筛选）**
   - 外部先根据预测 logits / 概率和阈值，对低置信度的 query 做过滤，
     例如将其 label 置为 ``ignore_label``；
   - 本模块在 ``update_memory`` 中只会根据类别 id
     更新对应 (batch_index, class_id) 的 FIFO 队列，
     对 ``ignore_label`` 以及越界类别（<0 或 >= num_classes）一律跳过；
   - 这样就满足了“**只对置信度大于阈值的 query 做 memory 更新**”的需求，
     同时保持接口与原实现兼容。

4. **读取规则**
   - 对于输入的 query tokens `cur_tokens_bqc`（B, Q, C）：
     - 对第 b 个样本，只会使用 **该样本自己的 class-wise memory**；
     - 实现方式是对每个 b 单独收集其所有类别的 memory，
       得到 `mem_b_LC`，再通过若干层 `QueryMemoryLayer`
       (self-attn + cross-attn + MLP) 做融合。

5. **编码部分保持与原 memory_encorder 一致**
   - 继续复用 `MaskTokenEncoder` + `TemporalEncoding` + `GatedFuser`，
     将 (query, mask) 编成 (B, Q, C) 的 token；
   - 你可以：
       - 用本模块的 ``encode_frame_tokens(...)`` 编码当前帧，
         然后：
           - 用 ``read_memory(...)`` 从本 batch 的 class-wise memory 里读取；
           - 再在外部根据 GT / 预测类别（已做置信度阈值过滤）
             调用 ``update_memory(...)``；
       - 或者直接把已经是 (B, Q, C) 的 query 特征送进
         ``read_memory`` / ``update_memory``。

本文件只负责 **按 batch 独立的 class-wise memory 的维护与读写逻辑**，
不直接耦合到 MaskFormer / tracker 的 loss 计算中，
方便你在外部灵活接入 GT / 匹配结果与置信度筛选逻辑。
"""

from .memory_encorder import (
    MaskTokenEncoder,
    TemporalEncoding,
    GatedFuser,
    QueryMemoryLayer,
)


class ClasswiseQueryMemoryModule(nn.Module):
    """按 **batch 独立、类别级 FIFO** 的 query-level memory 模块。

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
        - 本 FIFO 版本中，memory 在同一个 mini-batch 内按样本维度 b 独立；
        - 若你希望不同 mini-batch / 不同视频之间也不共享 memory，
          需要在合适的位置（例如每个 batch / 每个视频开始时）
          手动调用 ``reset_memory()``；
        - 可以通过 ``reset_memory()`` 在需要的时候清空所有 device 上的 memory。
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

        # 每个 device 维护一个两层 dict:
        #   device -> { batch_idx -> { class_id -> Tensor(K_c, C) } }
        #   - 最外层按 device 区分（兼容多 GPU / DDP 场景）；
        #   - 第二层按 batch 中的样本索引 b 区分，不同样本互不共享 memory；
        #   - 最内层按语义类别 class_id 区分，每个 queue 为一个 FIFO 队列：
        #       * 形状为 (K_c, C)，K_c <= per_class_max_queries；
        #       * 超过上限时丢弃最早进入的条目，实现先进先出。
        self._memory_bank: Dict[torch.device, Dict[int, Dict[int, torch.Tensor]]] = {}

        # 下面这个时间戳结构只在多样性更新分支中使用（若你以后需要），
        # 当前 FIFO 版本的 update_memory 不依赖它，但保留字段以兼容原接口。
        # 结构: device -> { class_id -> Tensor(K_c,) }
        self._memory_last_update: Dict[torch.device, Dict[int, torch.Tensor]] = {}

    # ------------------------------------------------------------------
    # 内部工具函数：bank 管理
    # ------------------------------------------------------------------
    def _get_device_bank(self, device: torch.device) -> Dict[int, Dict[int, torch.Tensor]]:
        """获取 / 初始化当前 device 上的整体 memory 结构。

        返回结构：``{batch_idx -> {class_id -> Tensor(K_c, C)}}``。
        """

        if device not in self._memory_bank:
            self._memory_bank[device] = {}
        return self._memory_bank[device]

    def _get_batch_bank(self, device: torch.device, batch_idx: int) -> Dict[int, torch.Tensor]:
        """获取 / 初始化某个 device + batch_idx 下的 class-wise memory。

        返回结构：``{class_id -> Tensor(K_c, C)}``。
        """

        dev_bank = self._get_device_bank(device)
        if batch_idx not in dev_bank:
            dev_bank[batch_idx] = {}
        return dev_bank[batch_idx]

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
    def _collect_batch_memory_tokens(self, device: torch.device, batch_idx: int) -> Optional[torch.Tensor]:
        """收集某个 device + batch_idx 下所有类别的 memory，并按类别 id 排序后拼接。

        返回：
            - mem_LC: (L_mem, C) 或
            - None: 若该样本目前没有任何 memory。
        """

        batch_bank = self._get_batch_bank(device, batch_idx)
        if not batch_bank:
            return None

        mem_list = []
        for cls_id in sorted(batch_bank.keys()):
            mem_cls = batch_bank[cls_id]
            if mem_cls.numel() == 0:
                continue
            # mem_cls: (K_c, C)
            mem_list.append(mem_cls)

        if len(mem_list) == 0:
            return None

        mem_LC = torch.cat(mem_list, dim=0)  # (L_mem, C)
        return mem_LC

    def read_memory(self, cur_tokens_bqc: torch.Tensor) -> torch.Tensor:
        """用当前 tokens (B,Q,C) 从 **本 batch 独立的 class-wise memory bank** 中读取信息。

        - 对第 b 个样本，只会使用该样本自己的 memory：
            1) 收集该样本下所有类别的 memory，得到 mem_LC_b (L_mem_b, C)；
            2) 通过若干层 QueryMemoryLayer 对 (1,Q,C) 与 (1,L_mem_b,C)
               做 self-attn + cross-attn + MLP 融合；
        - 若该样本没有任何 memory，则直接返回其原始 tokens。
        """

        dev = cur_tokens_bqc.device
        B, _, _ = cur_tokens_bqc.shape

        out_list = []
        for b in range(B):
            cur_bqc = cur_tokens_bqc[b : b + 1]  # (1,Q,C)
            mem_LC = self._collect_batch_memory_tokens(dev, b)
            if mem_LC is None:
                # 当前样本还没有 memory，直接返回自身
                out_list.append(cur_bqc)
                continue

            mem_bLc = mem_LC.unsqueeze(0).contiguous()  # (1,L_mem,C)

            x = cur_bqc
            for layer in self.layers:
                x = layer(x, mem_bLc)
            out_list.append(x)

        return torch.cat(out_list, dim=0)

    # ------------------------------------------------------------------
    # 更新：在模块外部根据 GT / 预测标签调用
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update_memory(
        self,
        tokens_bqc: torch.Tensor,     # (B,Q,C)
        class_labels_bq: torch.Tensor,  # (B,Q)，语义类别 id
        ignore_label: list = None,
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
                * 否则：丢弃最早进入的条目（FIFO），append 新 token。
        """

        assert tokens_bqc.shape[:2] == class_labels_bq.shape[:2], (
            f"tokens_bqc shape {tokens_bqc.shape} and class_labels_bq shape {class_labels_bq.shape} mismatch"
        )

        dev = tokens_bqc.device

        if self.detach_memory:
            tokens_bqc = tokens_bqc.detach()

        B, Q, C = tokens_bqc.shape

        # 逐个 (b, q) 进行更新，保证：
        #   - 不同 batch 样本使用独立的 memory；
        #   - 每个 (b, class_id) 下的队列采用 FIFO 策略；
        #   - ignore_label 或越界类别一律跳过。
        for b in range(B):
            batch_bank = self._get_batch_bank(dev, b)

            for q in range(Q):
                cls_id = int(class_labels_bq[b, q].item())

                # 过滤掉无效类别
                if cls_id < 0:
                    continue
                if ignore_label is not None and cls_id in ignore_label:
                    continue
                if cls_id < 0 or cls_id >= self.num_classes:
                    continue

                tok = tokens_bqc[b, q]  # (C,)

                if cls_id not in batch_bank:
                    # 该 (b, class_id) 目前还没有任何 memory，直接初始化
                    batch_bank[cls_id] = tok.unsqueeze(0).clone()  # (1, C)
                    continue

                mem_old = batch_bank[cls_id]  # (K_c, C)
                # 采用简单直观的 FIFO 策略：
                #   - 先把新 token 追加到队列尾部；
                #   - 若长度超过 per_class_max_queries，则丢弃最早进入的条目。
                mem_new = torch.cat([mem_old, tok.unsqueeze(0)], dim=0)  # (K_c+1, C)
                if mem_new.shape[0] > self.per_class_max_queries:
                    mem_new = mem_new[-self.per_class_max_queries :]

                batch_bank[cls_id] = mem_new


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
