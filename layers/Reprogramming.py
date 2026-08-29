"""P5：时序-语言对齐层（Reprogramming）。

设计要点（依据 execution_plan.md P5）：
- Q = MSTF 输出的 patch 嵌入（已是 768 维），K/V = 原型库软选择 + Value 聚合；
- 原型库两种来源：可学习 codebook（首选）/ 词嵌入候选子集（预筛选后 512×512 选择矩阵，禁止全词表）；
- 零初始化残差门控：Step 0 输出与未插入时严格一致（无损起点）。
"""
from math import sqrt

import torch
from torch import nn


class _CrossAttentionReprogram(nn.Module):
    """与 models/TimeLLM.py 的 ReprogrammingLayer 同构的 Q→(K,V) 软选择聚合。"""

    def __init__(self, d_in, n_heads, dropout=0.1):
        super().__init__()
        assert d_in % n_heads == 0, 'reprog_heads 必须整除 d_llm=768'
        self.d_keys = d_in // n_heads
        self.n_heads = n_heads
        self.query_projection = nn.Linear(d_in, d_in)
        self.key_projection = nn.Linear(d_in, d_in)
        self.value_projection = nn.Linear(d_in, d_in)
        self.out_projection = nn.Linear(d_in, d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, target_embedding, prototypes):
        # target_embedding: [B, L, d_in]；prototypes: [S, d_in]
        B, L, _ = target_embedding.shape
        S = prototypes.shape[0]
        H = self.n_heads
        E = self.d_keys

        q = self.query_projection(target_embedding).view(B, L, H, E)
        k = self.key_projection(prototypes).view(S, H, E)
        v = self.value_projection(prototypes).view(S, H, E)

        scale = 1.0 / sqrt(E)
        scores = torch.einsum('blhe,she->bhls', q, k)
        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        out = torch.einsum('bhls,she->blhe', attn, v)
        out = out.reshape(B, L, -1)
        return self.out_projection(out)


class TimeSeriesReprogramming(nn.Module):
    def __init__(self, d_in, num_prototypes=512, n_heads=8, proto_source='codebook',
                 word_embeddings=None, dropout=0.1):
        super().__init__()
        self.proto_source = proto_source
        if proto_source == 'codebook':
            self.codebook = nn.Parameter(torch.randn(num_prototypes, d_in) * 0.02)
        elif proto_source == 'word_embed':
            assert word_embeddings is not None, \
                'proto_source=word_embed 需要主干词嵌入（--no_backbone 下不可用）'
            with torch.no_grad():
                w = word_embeddings.detach().float()
                # 按行范数预筛选候选词（激活度代理），禁止全词表软选择
                _, idx = w.norm(dim=-1).topk(num_prototypes)
                candidates = w[idx].clone()
            self.register_buffer('proto_candidates', candidates)
            # 选择矩阵仅作用于候选子集索引维：prototypes = W @ candidates（[512,512]×[512,768]）
            self.proto_select_weight = nn.Parameter(torch.eye(num_prototypes))
        else:
            raise ValueError(f'未知 proto_source: {proto_source}')

        self.reprog = _CrossAttentionReprogram(d_in, n_heads, dropout)
        # 零初始化残差门控：tanh(0)=0，Step 0 输出恒等于输入
        self.residual_gate = nn.Parameter(torch.zeros(1))

    def forward(self, enc_out):
        in_dtype = enc_out.dtype
        if self.proto_source == 'codebook':
            prototypes = self.codebook
        else:
            prototypes = self.proto_select_weight.matmul(self.proto_candidates).to(in_dtype)
        aligned = self.reprog(enc_out.to(prototypes.dtype), prototypes)
        return enc_out + torch.tanh(self.residual_gate) * aligned.to(in_dtype)
