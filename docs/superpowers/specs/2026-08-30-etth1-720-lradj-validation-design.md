# ETTh1 pred_len=720 学习率调度验证设计

## 目标

在实际初始学习率均约为 `0.0004` 的条件下，方向性比较 COS 与当前 type1 的不同衰减轨迹。新 COS 结果可与既有 COS-0.01 日志作辅助对照，但只有在代码版本、完整参数和 seed 均一致时，才能进一步支持“原 COS 表现差主要由 `0.01` 过大导致”的因果判断。

## 修改范围

仅修改 `scripts/ETTh1_hybrid.sh`，不修改 `run_main.py`、模型代码、数据代码或学习率调度实现。

## 实验配置

脚本关闭当前启用的 `pred_len=336`，随后串行运行两组 `pred_len=720`：

1. COS：传入 `--learning_rate 0.0004 --lradj COS`，首轮实际学习率为 `0.0004`。
2. type1：传入 `--learning_rate 0.01 --lradj type1`。按当前 `run_main.py` 行为，`OneCycleLR` 构造时通过默认 `div_factor=25` 将首轮实际学习率设为约 `0.0004`。

前几轮实际训练 LR 时序为：

| epoch | COS | type1 |
|---:|---:|---:|
| 1 | `0.0004` | 约 `0.0004` |
| 2 | 约 `0.00039990` | 约 `0.0004` |
| 3 | 约 `0.00039961` | 约 `0.0002` |

type1 在 epoch 1 结束时先把 `args.learning_rate` 覆盖为约 `0.0004`，再写回相同值，因此到 epoch 3 才首次减半。

除 `learning_rate`、`lradj` 和用于区分结果的 `model_comment` 外，两组训练参数保持一致。模型注释分别使用：

- `hybrid_720_cos_lr4e4`
- `hybrid_720_type1_nominal_lr1e2`

## 执行与日志

脚本首先根据自身路径切换到仓库根目录，避免调用者当前目录改变数据、代码和日志解析结果。

两组实验写入同一个精确到秒并包含进程号的会话日志，避免短时间重跑追加到同一文件。启动前预检日志与结构化结果目录可写，并记录 `run_main.py`、脚本自身的 SHA-256、固定 seed、完整展开参数，以及每组明确的 START/END 标记和进程退出码。每组启动前打印调度器、传入学习率及预期实际初始学习率，便于核验实验契约。

两组分别通过 `--result_json` 写入唯一的结构化结果文件；结果路径是审计参数，不属于训练变量。脚本继续使用 `set -eo pipefail`：若第一组 COS 进程以非零状态退出，脚本立即停止；进程正常返回后，再同步解析 COS 的结构化状态。只有所有 run 的状态均为 `completed` 或 `early_stopped` 时才启动 type1；若结果文件缺失、JSON 无法解析、包含 `[Abort]` 对应的 `nonfinite_*` 等其他状态，则打印失败 END 标记并以非零状态退出。本次不修改 `run_main.py` 的退出语义。

## 验证标准

修改后仅执行静态检查，不启动训练：

- `bash -n scripts/ETTh1_hybrid.sh` 必须通过。
- 静态确认只有两条未注释的 `accelerate launch` 命令。
- 两条命令的 `pred_len` 都为 `720`。
- 除已声明的三个差异字段外，两组训练参数一致；两个不同的 `result_json` 路径仅作为审计字段豁免。
- 通过不启动训练的 argv 级解析检查两条完整命令，确认顺序、flag/value 配对、无续行断裂产生的裸 `--...` 命令，并确认规范化参数差异严格为 `learning_rate`、`lradj`、`model_comment`，另仅允许审计字段 `result_json` 的值不同。
- 静态检查 COS 与 type1 各自使用唯一结果路径，并确认两组之间存在 COS 状态门禁。
- 实现前后 `run_main.py` 的 SHA-256 必须保持为 `8c24dca68bd0d38d97061f28c7869066d27e6e30d7efce880900c2a157c3053f`。

本次保持默认 `save_checkpoint=False` 和 `es_mode=ema`。结果报告采用日志中 raw Validation 最低轮次的配对 Test 指标；它不是恢复 checkpoint 后重新测试的结果，不称为“checkpoint 指标”，也不使用跨 epoch 的最佳 Test 值。
