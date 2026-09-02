"""
层次化动态 Prompt 生成器
=========================

三层结构，共16个token，全部在连续 embedding 空间生成：
  - Layer 1: Meta tokens    (4个) — 任务级信息，可学习基底 + FiLM 调制
  - Layer 2: Pattern tokens (6个) — 模式原型库 + sparsemax 路由
  - Layer 3: Numerical tokens(6个) — 独立 MLP 编码统计特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NumericalTokenEncoder(nn.Module):
    """
    数值层编码器：每个统计维度有独立的 MLP，
    生成对应的 token embedding。
    """

    def __init__(self, d_stat, d_llm, num_tokens=6):
        super().__init__()
        self.num_tokens = num_tokens

        # 每个 token 有独立的编码网络
        self.token_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_stat, 128),
                nn.GELU(),
                nn.Linear(128, d_llm),
            )
            for _ in range(num_tokens)
        ])

        # 位置偏置：让不同 token 在 embedding 空间有基础区分
        self.position_bias = nn.Parameter(
            torch.randn(num_tokens, d_llm) * 0.01
        )

    def forward(self, s):
        """
        Args:
            s: [B, d_stat] 统计特征
        Returns:
            tokens: [B, num_tokens, d_llm]
        """
        tokens = []
        for i, encoder in enumerate(self.token_encoders):
            t = encoder(s) + self.position_bias[i]  # [B, d_llm]
            tokens.append(t.unsqueeze(1))
        return torch.cat(tokens, dim=1)  # [B, num_tokens, d_llm]


class HierarchicalDynamicPrompt(nn.Module):
    """
    层次化动态 Prompt 生成器

    设计哲学：模拟人类专家从宏观到微观、定性到定量地描述时序的方式，
    但在连续 embedding 空间完成。

    三层结构：
      - Meta tokens    (4个)：这段序列"是什么" — 任务级先验 + 轻微样本调制
      - Pattern tokens (6个)：这段序列"长什么样" — 统计特征驱动的模式路由
      - Numerical tokens(6个)：这段序列"量级如何" — 直接编码数值统计量
    """

    def __init__(
        self,
        d_stat=12,
        d_llm=768,
        num_meta=4,
        num_pattern=6,
        num_numerical=6,
        num_pattern_types=8,
    ):
        super().__init__()
        self.num_meta = num_meta
        self.num_pattern = num_pattern
        self.num_numerical = num_numerical
        self.d_llm = d_llm

        # ── Layer 1：元信息层 ──────────────────────────────
        # 可学习的任务级 token（跨样本共享部分）
        self.meta_base = nn.Parameter(
            torch.randn(num_meta, d_llm) * 0.02
        )
        # FiLM 调制：用统计特征生成 scale 和 shift
        self.meta_modulator = nn.Sequential(
            nn.Linear(d_stat, num_meta * 2),
        )

        # ── Layer 2：模式层 ────────────────────────────────
        # 模式原型库（8 种基础模式）
        self.pattern_prototypes = nn.Parameter(
            torch.randn(num_pattern_types, num_pattern, d_llm) * 0.02
        )
        # 路由网络：统计特征 → 模式权重
        self.pattern_router = nn.Sequential(
            nn.Linear(d_stat, 64),
            nn.GELU(),
            nn.Linear(64, num_pattern_types),
        )

        # ── Layer 3：数值层 ────────────────────────────────
        self.numerical_encoder = NumericalTokenEncoder(
            d_stat=d_stat,
            d_llm=d_llm,
            num_tokens=num_numerical,
        )

        # 层间归一化
        self.norm = nn.LayerNorm(d_llm)

    def forward(self, s):
        """
        Args:
            s: [B, d_stat] 统计特征向量
        Returns:
            prompt_tokens: [B, 16, d_llm]
            info: dict，包含中间结果供可视化 / 辅助损失
        """
        B = s.shape[0]

        # ── Layer 1：元信息 tokens ─────────────────────────
        meta_base = self.meta_base.unsqueeze(0).expand(B, -1, -1)  # [B, 4, d_llm]

        # bf16 的 scale 可能恰好变成 -1，使整个 meta token 经 LN 后归零；
        # 该零 token 会在冻结 Mamba 的多层 RMSNorm 中放大反向梯度。
        # 仅此小型 FiLM 路径使用 FP32，其余模块继续遵循外层 autocast。
        with torch.autocast(device_type=s.device.type, enabled=False):
            mod = self.meta_modulator(s.float())   # [B, num_meta * 2]
            scale, shift = mod.chunk(2, dim=-1)    # [B, num_meta] each
            scale = scale.unsqueeze(-1)           # [B, num_meta, 1]
            shift = shift.unsqueeze(-1)           # [B, num_meta, 1]
            meta_tokens = meta_base * (1 + scale) + shift  # [B, 4, d_llm]

        # ── Layer 2：模式 tokens ───────────────────────────
        router_logits = self.pattern_router(s)     # [B, num_pattern_types]
        router_weights = self._sparsemax(router_logits)  # [B, num_pattern_types]

        # 软路由：加权组合模式原型
        pattern_tokens = torch.einsum(
            'bn,npd->bpd',
            router_weights,
            self.pattern_prototypes,
        )  # [B, 6, d_llm]

        # ── Layer 3：数值 tokens ───────────────────────────
        numerical_tokens = self.numerical_encoder(s)  # [B, 6, d_llm]

        # ── 拼接三层 ──────────────────────────────────────
        prompt_tokens = torch.cat(
            [meta_tokens, pattern_tokens, numerical_tokens],
            dim=1,
        )  # [B, 16, d_llm]

        return self.norm(prompt_tokens), {
            'router_weights': router_weights,
            'meta_tokens': meta_tokens,
            'pattern_tokens': pattern_tokens,
            'numerical_tokens': numerical_tokens,
        }

    def _sparsemax(self, logits):
        """
        Top-3 Softmax：比全 softmax 更稀疏，
        让路由集中到 1-3 个主要模式。
        """
        topk_val, topk_idx = logits.topk(3, dim=-1)
        topk_w = torch.softmax(topk_val, dim=-1)
        # 使用 topk_w 的 dtype 创建 weights，避免 bf16 autocast 下的类型不匹配
        weights = torch.zeros(logits.shape, dtype=topk_w.dtype, device=logits.device)
        weights.scatter_(-1, topk_idx, topk_w)
        return weights
