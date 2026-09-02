# TimeMamba B4 正式基线 Worktree 设计

日期：2026-09-02  
源工作树：`/home/Lain/Code/TimeMamba/0823/timemamba-explore`  
目标工作树：`/home/Lain/Code/TimeMamba/0823/timemamba-baseline`  
目标分支：`baseline`

## 1. 目标

创建一个独立、干净、可运行的 Git worktree，把已经通过三 seed、Val-only 验证的 B4 模型提升为新的正式主基线。目标工作树用于后续逐个接入新论文模块，不携带当前探索工作树中的其他消融副本、日志、结果或临时诊断文件。

B4 的功能定义固定为：以修正后的 P0 为基础，只删除 HierarchicalDynamicPrompt 的 pattern prototypes、pattern router 和 6 个 pattern tokens；保留 meta FiLM、numerical tokens、small/large patch、FFT、冻结 Mamba 和原预测头。

## 2. Git 与来源策略

1. 创建前记录并固定当前已审查的 `explore` HEAD（`git rev-parse HEAD`），再次确认目标目录、目标分支均不存在，然后仅执行一次 `git worktree add -b baseline /home/Lain/Code/TimeMamba/0823/timemamba-baseline <固定的base-sha>`。禁止在 dirty 源工作树切换分支。
2. 以 B4 正式复验快照 `results/b_ablation/source_snapshot_b4_seedrep_20260902T130644/` 作为运行代码真值，不从当前大量 dirty/untracked 文件整体复制。
3. 快照自带的 `SHA256SUMS` 错误地包含其自身，完整 `sha256sum -c` 会仅在该自引用条目失败。迁移前应以 `manifest.json.source_sha256` 逐项校验 11 个源文件，并对 `SHA256SUMS` 排除自身条目后复核；该已知归档缺陷不写回历史快照。
4. 在新分支提交一份独立的“B4 正式基线”实现提交；所有复制、暂存和提交命令都以目标路径为工作目录，只暂存规格列出的白名单文件。最终汇报分别列出源分支的规格提交链、固定 base SHA 和新分支实现提交。
5. 已提交基点包含 `logs/ETTh1/ETTh1_96_main_experiment_2026-08-29-23-39.log`。实现提交必须删除该跟踪日志，并在 `.gitignore` 中忽略 `logs/`、`results/` 和 `checkpoints/`，保证目标不携带历史实验产物。
6. 新 worktree 完成后保持 Git 工作树干净；本地 `dataset` 符号链接受 `.gitignore` 的 `dataset/` 规则保护，不进入提交。

## 3. 正式入口形态

新 worktree 不保留 B4 作为外部实验入口，而是提升为正式主入口：

- `models/TimeMamba_b4.py` 的验证逻辑迁移到 `models/TimeMamba.py`；
- `run_b4.py` 的验证训练逻辑迁移到 `run_main.py`，导入主 `models.TimeMamba`；
- `scripts/ETTh1.sh` 不使用未归档的 `scripts/ETTh1_b4.sh`。它从正式快照中的 `scripts/ETTh1_b4_seedrep.sh` 提取相同的四 horizon、LR、lradj 和公共参数，改为 seed 2025 并直接调用 `run_main.py`；正式基线不依赖 `tools/run_with_grad_monitor.py`；
- 内部 `_B4Prompt` 等实验命名改为中性的 baseline 命名；不保留旧名称别名或兼容分支；
- 删除 `variant=p0/a0/a1`、A0/A1 feature projection、residual alpha 及其 CLI 分支；固定使用 B4/P0 的前 64 隐维读出。删除无效的外部 `prompt_tokens` 与 `num_pattern_types` 参数，但在模型内部固定按 `4 meta + 6 pattern + 6 numerical、8 prototypes` 完整构造后删除 pattern 参数，以保留已验证 B4 的 RNG 消耗和初始化；
- `baseline.md` 更新为新基线的来源、结构、配置、Val-only 证据和限制说明。

共享文件从正式 B4 快照迁移：

- `data_provider/data_factory.py`
- `data_provider/data_loader.py`
- `layers/Embed.py`
- `layers/HierarchicalPrompt.py`
- `layers/StandardNorm.py`
- `utils/tools.py`

源工作树未跟踪的 `AGENTS.md` 属于项目执行约束而非实验产物，应作为白名单文件迁入并提交，使新 worktree 独立保留 `time-llm` 环境、中文回复和称呼 Lain 等规则。

其余仓库文件沿用已提交 HEAD，除非运行或测试证明缺少 B4 必需依赖；不得顺手迁移 A/B 其他实验文件。

## 4. 数据与产物边界

- 新 worktree 的 `dataset` 指向源工作树 `/home/Lain/Code/TimeMamba/0823/timemamba-explore/dataset`，使用本地绝对符号链接。
- 不复制 `logs/`、`results/`、`checkpoints/`、`.zcode/`、A0/A1/B1–B7 文件或 Deep Research 临时产物。
- 运行日志和结果由新 worktree 后续实验自行生成。
- 不修改或删除源工作树中的任何用户文件。

## 5. 验证标准

### 静态验证

- `git status --short`：提交后为空；仅创建 dataset 链接时仍应因 ignore 规则保持为空。
- `bash -n scripts/ETTh1.sh` 通过。
- Python 模块可编译、主入口 `--help` 可执行。
- 主脚本显式包含四个 horizon 和既定 LR/lradj。

### B4 等价性

建立可复现等价性夹具，固定 `variant=p0, seq_len=96, pred_len=96, d_ff=64, d_model=32, dropout=0.2, llm_layers=24, prompt_tokens=16, num_pattern_types=8, enc_in=7, small_patch/stride=8/4, large_patch/stride=24/4`。在两个隔离 Python 解释器或无模块缓存污染的独立导入空间中分别加载正式 B4 快照和目标主模型；每次构造前重置 Python、NumPy、PyTorch CPU/CUDA RNG。比较：

- state dict 键集合相同；
- 所有公共参数逐位一致；
- h96 可训练参数量为 2,641,250；h192/h336/h720 分别为 `2,788,802 / 3,010,130 / 3,600,338`；
- pattern prototypes/router 不存在；
- meta 和 numerical 参数存在；
- eval 模式下使用固定、相同输入，输出 `torch.equal`；若底层 GPU kernel 不保证逐位确定，则至少要求严格数值近似并披露最大绝对差，不能静默降级；
- 输出形状为 `[B, pred_len, 7]` 且有限。

### 训练 smoke

在 `time-llm` 环境、真实 ETTh1 batch 和 GPU 上执行至少一个优化步骤：

- bf16 前向有限；
- loss 与梯度有限；
- 无非预期 `grad=None`；
- 冻结 Mamba 无梯度；
- Adam step 后参数有限。

若 GPU 被无关任务占用，不杀进程；完成静态/CPU检查后报告 GPU smoke 未执行的阻塞状态。

## 6. 交付状态

最终汇报必须包含：目标路径、分支、规格提交链、固定 base SHA、新分支实现提交、B4来源快照、迁移文件清单、被删除的历史跟踪日志、dataset链接、Git干净状态、测试结果及任何未完成验证。不得把 B4 描述为所有数据集或所有 horizon 上普遍优于 P0；它是当前 ETTh1、四 horizon 等权 Val-only 口径下的最简可靠基线。
