# TimeMamba B4 正式基线 Worktree 设计

日期：2026-09-02  
源工作树：`/home/Lain/Code/TimeMamba/0823/timemamba-explore`  
目标工作树：`/home/Lain/Code/TimeMamba/0823/timemamba-baseline`  
目标分支：`codex/timemamba-baseline`

## 1. 目标

创建一个独立、干净、可运行的 Git worktree，把已经通过三 seed、Val-only 验证的 B4 模型提升为新的正式主基线。目标工作树用于后续逐个接入新论文模块，不携带当前探索工作树中的其他消融副本、日志、结果或临时诊断文件。

B4 的功能定义固定为：以修正后的 P0 为基础，只删除 HierarchicalDynamicPrompt 的 pattern prototypes、pattern router 和 6 个 pattern tokens；保留 meta FiLM、numerical tokens、small/large patch、FFT、冻结 Mamba 和原预测头。

## 2. Git 与来源策略

1. 从当前 `explore` 分支的已提交 HEAD 建立 `codex/timemamba-baseline`。
2. 以 B4 正式复验快照 `results/b_ablation/source_snapshot_b4_seedrep_20260902T130644/` 作为运行代码真值，不从当前大量 dirty/untracked 文件整体复制。
3. 在新分支提交一份独立的“B4 正式基线”提交；不把源工作树其他用户改动加入该提交。
4. 新 worktree 完成后保持 Git 工作树干净；本地 `dataset` 符号链接受 `.gitignore` 的 `dataset/` 规则保护，不进入提交。

## 3. 正式入口形态

新 worktree 不保留 B4 作为外部实验入口，而是提升为正式主入口：

- `models/TimeMamba_b4.py` 的验证逻辑迁移到 `models/TimeMamba.py`；
- `run_b4.py` 的验证训练逻辑迁移到 `run_main.py`，导入主 `models.TimeMamba`；
- `scripts/ETTh1_b4.sh` 的四 horizon 协议整理到 `scripts/ETTh1.sh`，显式记录 seed 2025；
- 内部 `_B4Prompt` 等实验命名改为中性的 baseline 命名；不保留旧名称别名或兼容分支；
- `baseline.md` 更新为新基线的来源、结构、配置、Val-only 证据和限制说明。

共享文件从正式 B4 快照迁移：

- `data_provider/data_factory.py`
- `data_provider/data_loader.py`
- `layers/Embed.py`
- `layers/HierarchicalPrompt.py`
- `layers/StandardNorm.py`
- `utils/tools.py`

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

使用同一 seed 和真实配置比较正式 B4 快照与新主模型：

- state dict 键集合相同；
- 所有公共参数逐位一致；
- 可训练参数量为 2,641,250；
- pattern prototypes/router 不存在；
- meta 和 numerical 参数存在；
- eval 模式固定输入输出一致；
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

最终汇报必须包含：目标路径、分支、两个新增提交、B4来源快照、迁移文件清单、dataset链接、Git干净状态、测试结果及任何未完成验证。不得把 B4 描述为所有数据集或所有 horizon 上普遍优于 P0；它是当前 ETTh1、四 horizon 等权 Val-only 口径下的最简可靠基线。
