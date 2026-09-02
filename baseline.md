# TimeMamba Minimal Baseline

本分支将已验证的 B4 结构作为正式主基线。它建立在修正后的 P0 上，唯一结构删除是 HierarchicalDynamicPrompt 的 pattern prototypes、pattern router 和 6 个 pattern tokens；meta FiLM 与 numerical tokens 保留。

## 固定结构

- ETTh1 七变量整窗输入，模型内部仍按 `B*N` channel-independent 处理；
- prompt 为 4 个 meta tokens + 6 个 numerical tokens；
- small patch `8/4`、large patch `24/4` 与 FFT 幅值分支保留；
- 冻结 24 层 Mamba-130M，固定取前 64 个隐藏维；
- raw Validation loss 早停，patience 3；
- 每轮计算 Test，最终报告最低 Val 轮次对应的 Test；模块选择采用 Val-only。

## B4 证据

ETTh1、seed 2025/2026/2027、pred_len 96/192/336/720 等权平均下，B4 相对 P0 的 Avg Validation loss 配对差值为：

- seed 2025：`-0.0021385`
- seed 2026：`-0.0041025`
- seed 2027：`-0.0039473`
- mean ± population std：`-0.0033961 ± 0.0008915`

可训练参数在 h96 配置下由 2,679,466 降至 2,641,250。该结果支持把本结构作为当前 ETTh1 实验范围内的最简可靠基线，不证明 pattern 机制在所有数据集或所有 horizon 上普遍无效。

## 来源

运行真值来自源工作树归档：

`results/b_ablation/source_snapshot_b4_seedrep_20260902T130644/`

正式入口：`scripts/ETTh1.sh -> run_main.py -> models/TimeMamba.py`。
