"""TimeMamba minimal baseline：保留 meta/numerical prompt，删除 pattern 层。

初始化时仍按已验证的 P0 顺序构造完整 prompt，再删除 pattern 参数，以保持
正式 B4 实验的公共参数初始化。模型固定使用前 64 个 Mamba 隐藏维读出。
"""
from math import sqrt
import os
from pathlib import Path
import torch.nn.functional as F
import torch
import torch.nn as nn
from modelscope import snapshot_download  # 适配国内 ModelScope 下载
from transformers import MambaModel
from layers.Embed import PatchEmbedding
from layers.StandardNorm import Normalize
from layers.HierarchicalPrompt import HierarchicalDynamicPrompt


class _BaselinePrompt(HierarchicalDynamicPrompt):
    """正式基线 prompt：只输出 meta 与 numerical tokens。"""

    def forward(self, s):
        B = s.shape[0]
        # ── meta tokens（与 P0 相同的 FP32 FiLM 路径）──
        meta_base = self.meta_base.unsqueeze(0).expand(B, -1, -1)
        with torch.autocast(device_type=s.device.type, enabled=False):
            mod = self.meta_modulator(s.float())
            scale, shift = mod.chunk(2, dim=-1)
            scale = scale.unsqueeze(-1)
            shift = shift.unsqueeze(-1)
            meta_tokens = meta_base * (1 + scale) + shift
        numerical_tokens = self.numerical_encoder(s)
        tokens = torch.cat([meta_tokens, numerical_tokens], dim=1)
        return self.norm(tokens), {
            'meta_tokens': meta_tokens,
            'numerical_tokens': numerical_tokens,
        }
import transformers

# 屏蔽冗余的警告信息
transformers.logging.set_verbosity_error()


def resolve_modelscope_model_dir(model_id):
    owner, name = model_id.split('/', 1)
    cache_roots = []

    env_cache = os.getenv('MODELSCOPE_CACHE')
    if env_cache:
        cache_roots.append(Path(env_cache))
    cache_roots.append(Path.home() / '.cache' / 'modelscope')
    cache_roots.append(Path.home() / '.cache' / 'modelscope' / 'hub')

    candidate_suffixes = [
        Path('models') / owner / name,
        Path(owner) / name,
        Path('hub') / 'models' / owner / name,
    ]

    def is_valid_model_dir(candidate):
        return (
            candidate.is_dir()
            and (candidate / 'config.json').exists()
            and ((candidate / 'model.safetensors').exists()
                 or (candidate / 'pytorch_model.bin').exists())
        )

    for cache_root in cache_roots:
        for suffix in candidate_suffixes:
            candidate = cache_root / suffix
            if is_valid_model_dir(candidate):
                return str(candidate)
    return None

class FlattenHead(nn.Module):
    """
    预测头：将 LLM 输出的隐藏状态特征压扁并映射到预测窗口长度
    """

    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        # x shape: [Batch * n_vars, d_ff, patch_nums]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class MultiScaleTimeFreqFusion(nn.Module):
    """
    多尺度时频融合模块（MSTF, Multi-Scale Time-Frequency Fusion）

    三分支并行编码 + 特征维度融合：
      - 分支 1：小尺度 Patch → 门控卷积适配 → [B*N, patch_nums, d_llm]
      - 分支 2：大尺度 Patch → 门控卷积适配 → [B*N, patch_nums, d_llm]
      - 分支 3：频域（FFT 幅值谱）→ [B*N, patch_nums, d_llm]

    三路输出在特征维度拼接后经线性投影融合为统一表示。
    """

    def __init__(self, seq_len, d_model, d_llm,
                 small_patch, small_stride,
                 large_patch, large_stride,
                 dropout=0.1):
        super().__init__()

        # ── 小尺度 Patch 嵌入 ──
        self.patch_embedding_small = PatchEmbedding(
            d_model, small_patch, small_stride, dropout)
        patch_nums_ref = (seq_len + small_stride - small_patch) // small_stride + 1

        # ── 大尺度 Patch 嵌入 ──
        self.patch_embedding_large = PatchEmbedding(
            d_model, large_patch, large_stride, dropout)
        patch_nums_large = (seq_len + large_stride - large_patch) // large_stride + 1

        if patch_nums_ref != patch_nums_large:
            self.need_token_align = True
            self.large_token_align = nn.Linear(patch_nums_large, patch_nums_ref)
        else:
            self.need_token_align = False

        # ── 适配器（d_model → d_llm） ──
        self.adapter_conv = nn.Conv1d(
            d_model, d_model, 3, padding=1, groups=d_model)
        self.adapter_conv_norm = nn.LayerNorm(d_model)
        self.adapter_up = nn.Linear(d_model, d_llm)
        self.adapter_gate = nn.Linear(d_model, d_llm)
        self.adapter_norm = nn.LayerNorm(d_llm)
        self.adapter_drop = nn.Dropout(dropout)

        # ── 频域分支 ──
        freq_bins = seq_len // 2 + 1
        self.freq_token_align = nn.Linear(freq_bins, patch_nums_ref)
        self.freq_token_norm = nn.LayerNorm(patch_nums_ref)
        self.freq_feat_expand = nn.Linear(1, d_model)
        
        self.freq_conv = nn.Conv1d(
            d_model, d_model, 3, padding=1, groups=d_model)
        self.freq_conv_norm = nn.LayerNorm(d_model)
        self.freq_up = nn.Linear(d_model, d_llm)
        self.freq_gate = nn.Linear(d_model, d_llm)
        self.freq_out_norm = nn.LayerNorm(d_llm)
        self.freq_drop = nn.Dropout(dropout)

        # ── 融合投影 ──
        self.fusion_proj = nn.Linear(3 * d_llm, d_llm)

        self.patch_nums = patch_nums_ref

    def _gated_conv_adapt(self, x):
        """共享门控卷积适配：[B*N, patch_nums, d_model] → [B*N, patch_nums, d_llm]"""
        residual = x
        x = self.adapter_conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.adapter_conv_norm(F.silu(x) + residual)
        return self.adapter_drop(
            self.adapter_norm(self.adapter_up(x) * F.silu(self.adapter_gate(x))))

    def _freq_encode(self, x):
        """频域编码：[B, N, T] → [B*N, patch_nums, d_llm]"""
        B, N, T = x.shape
        x_amp = torch.fft.rfft(x.reshape(B * N, T), dim=-1).abs()

        x_tok = F.silu(self.freq_token_norm(self.freq_token_align(x_amp)))
        x_feat = self.freq_feat_expand(x_tok.unsqueeze(-1))

        residual = x_feat
        x_feat = self.freq_conv(x_feat.transpose(1, 2)).transpose(1, 2)
        x_feat = self.freq_conv_norm(F.silu(x_feat) + residual)
        out = self.freq_up(x_feat) * F.silu(self.freq_gate(x_feat))
        
        return self.freq_drop(self.freq_out_norm(out))

    def forward(self, x):
        """
        Args:
            x: [B, N, T] 归一化后的时间序列（channel-first）
        Returns:
            enc_out: [B*N, patch_nums, d_llm] 融合后的表示
            n_vars: 变量数 N
        """
        branches = []
        n_vars = None

        # 分支 1：小尺度 Patch → 适配
        enc_out_small, n_vars = self.patch_embedding_small(x.to(torch.bfloat16))
        enc_out_small = self._gated_conv_adapt(enc_out_small)
        branches.append(enc_out_small)

        # 分支 2：大尺度 Patch
        enc_out_large, n_vars_large = self.patch_embedding_large(x.to(torch.bfloat16))
        enc_out_large = self._gated_conv_adapt(enc_out_large)
        if n_vars is None:
            n_vars = n_vars_large

        if self.need_token_align:
            enc_out_large = self.large_token_align(
                enc_out_large.transpose(1, 2)
            ).transpose(1, 2)
        branches.append(enc_out_large)

        # 分支 3：频域
        enc_out_freq = self._freq_encode(x.float()).to(branches[0].dtype)
        branches.append(enc_out_freq)

        enc_out = torch.cat(branches, dim=-1)
        enc_out = self.fusion_proj(enc_out)

        return enc_out, n_vars


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5

        # --- 1. 初始化主干 ---
        model_id = 'AI-ModelScope/mamba-130m-hf'
        model_dir = resolve_modelscope_model_dir(model_id)
        if model_dir is None:
            snapshot_kwargs = {}
            modelscope_cache = os.getenv('MODELSCOPE_CACHE')
            if modelscope_cache:
                snapshot_kwargs['cache_dir'] = modelscope_cache
            model_dir = snapshot_download(model_id, **snapshot_kwargs)

        self.llm_model = MambaModel.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        self.d_llm = self.llm_model.config.hidden_size

        if hasattr(configs, 'llm_layers') and configs.llm_layers < len(self.llm_model.layers):
            self.llm_model.layers = self.llm_model.layers[:configs.llm_layers]

        for param in self.llm_model.parameters():
            param.requires_grad = False

        # 3. 层次化动态 Prompt
        self.d_stat = 12
        # 固定使用正式 B4 配置。pattern 参数先构造、后删除，只为保持已验证的
        # 公共参数初始化随机数流；它们不属于最终模型参数或前向路径。
        self.hierarchical_prompt = _BaselinePrompt(
            d_stat=self.d_stat,
            d_llm=self.d_llm,
            num_meta=4,
            num_pattern=6,
            num_numerical=6,
            num_pattern_types=8,
        )
        del self.hierarchical_prompt.pattern_prototypes
        del self.hierarchical_prompt.pattern_router

        # ============================================================
        # 多尺度时频融合模块（MSTF）
        # ============================================================
        self.mstf = MultiScaleTimeFreqFusion(
            seq_len=configs.seq_len,
            d_model=configs.d_model,
            d_llm=self.d_llm,
            small_patch=configs.small_patch,
            small_stride=configs.small_stride,
            large_patch=configs.large_patch,
            large_stride=configs.large_stride,
            dropout=configs.dropout,
        )

        self.patch_nums = self.mstf.patch_nums
        
        self.head_nf = self.d_ff * self.patch_nums

        self.output_projection = FlattenHead(
            n_vars=configs.enc_in,
            nf=self.head_nf,
            target_window=self.pred_len,
            head_dropout=configs.dropout,
        )

        self.normalize_layers = Normalize(1, affine=True)

    def readout_tokens(self, hidden):
        """仅处理最后 P 个数值 tokens；不混入 prompt 位置。"""
        hidden = hidden[:, -self.patch_nums:, :]
        return hidden[:, :, :self.d_ff]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, return_prompt_info=False):
        if return_prompt_info:
            dec_out, prompt_info = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, return_prompt_info=True)
            return dec_out[:, -self.pred_len:, :], prompt_info
        else:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, return_prompt_info=False):
        B, T, N = x_enc.size()
        x_enc = self.normalize_layers(x_enc, 'norm')

        # ============================================================
        # B. 层次化动态 Prompt
        # ============================================================
        x_flat = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T)  # [B*N, T]
        stats = self._compute_statistics(x_flat)  # [B*N, d_stat]
        prompt_embeddings, prompt_info = self.hierarchical_prompt(stats.to(torch.float32))

        # ============================================================
        # 多尺度时频融合（MSTF）
        # ============================================================
        x_enc_p = x_enc.permute(0, 2, 1).contiguous()  # [B, N, T]
        enc_out, n_vars = self.mstf(x_enc_p)

        # D. 喂给 Mamba
        prompt_embeddings = prompt_embeddings.to(enc_out.dtype)
        combined_embeddings = torch.cat([prompt_embeddings, enc_out], dim=1)
        
        mamba_out = self.llm_model(inputs_embeds=combined_embeddings).last_hidden_state

        # E. 输出处理与投影
        dec_out = self.readout_tokens(mamba_out)
        dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        # F. 反标准化
        out = self.normalize_layers(dec_out, 'denorm')
        if return_prompt_info:
            return out, prompt_info
        return out

    # ================================================================
    # 统计特征计算（实例归一化 + 12 维特征）
    # ================================================================
    def _compute_statistics(self, x):
        """
        从序列中提取 12 维归一化统计特征。
        先对每个样本做实例归一化，消除跨数据集量纲差异。

        Args:
            x: [B*N, T]
        Returns:
            stats: [B*N, 12]
        """
        # 实例归一化
        x_mean = x.mean(dim=-1, keepdim=True)
        x_std = x.std(dim=-1, keepdim=True) + 1e-8
        x_norm = (x - x_mean) / x_std

        stats = [
            x_norm.min(dim=-1).values,                           # 1. 归一化最小值
            x_norm.max(dim=-1).values,                           # 2. 归一化最大值
            x_norm.median(dim=-1).values,                        # 3. 归一化中位数
            x_norm.mean(dim=-1),                                 # 4. 归一化均值 (≈0)
            x_norm.std(dim=-1),                                  # 5. 归一化标准差 (≈1)
            self._linear_slope(x_norm),                          # 6. 线性趋势斜率
            self._skewness(x_norm),                              # 7. 偏度
            self._kurtosis(x_norm),                              # 8. 峰度
            self._autocorr_lag(x_norm, lag=1),                   # 9. lag-1 自相关
            self._autocorr_lag(x_norm, lag=2),                   # 10. lag-2 自相关
            x_norm.diff(dim=-1).std(dim=-1),                     # 11. 波动率（差分标准差）
            x_norm.max(dim=-1).values - x_norm.min(dim=-1).values,  # 12. 极差
        ]
        return torch.stack(stats, dim=-1)  # [B*N, 12]

    def _linear_slope(self, x):
        """可微的线性回归趋势斜率。x: [B, T]"""
        T = x.shape[-1]
        t = torch.linspace(0, 1, T, device=x.device, dtype=x.dtype).unsqueeze(0)  # [1, T]
        t_mean = 0.5
        x_mean = x.mean(dim=-1, keepdim=True)
        slope = ((t - t_mean) * (x - x_mean)).sum(dim=-1) / \
                ((t - t_mean).pow(2).sum() + 1e-8)
        return slope  # [B]

    def _skewness(self, x):
        """样本偏度。x: [B, T]"""
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-8
        return ((x - mean) / std).pow(3).mean(dim=-1)  # [B]

    def _kurtosis(self, x):
        """样本峰度（excess kurtosis）。x: [B, T]"""
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-8
        return ((x - mean) / std).pow(4).mean(dim=-1) - 3.0  # [B]

    def _autocorr_lag(self, x, lag=1):
        """lag-k 自相关系数。x: [B, T]"""
        mean = x.mean(dim=-1, keepdim=True)
        x_centered = x - mean
        var = (x_centered ** 2).sum(dim=-1) + 1e-8
        cov = (x_centered[:, :-lag] * x_centered[:, lag:]).sum(dim=-1)
        return cov / var  # [B]
