# TimeMamba 模块实验总结文档设计

日期：2026-09-01
目标文件：`docs/module_experiment_summary.md`

## 1. 目标

建立一份面向后续模块筛选的统一总结文档，先完整收录 A0、A1，未来可以用同一结构追加 A2、B0 等实验。文档必须回答：模块改了什么、修改前后四个 horizon 与平均指标如何变化、正式日志和结构化结果在哪里、当前证据支持什么结论。

## 2. 数据与口径

- 模型为 TimeMamba，数据为 ETTh1，`seq_len=96`，seed=2025。
- 每个 horizon 取最低**原始 Validation loss**轮次对应的 Test MSE/MAE，再对96、192、336、720等权平均。
- 不按最低 Test 选轮，不把原入口描述成恢复 checkpoint 后 Test-once。
- P0、A0、A1均为独立从头训练；A1不继承A0训练权重。
- 结果是单seed方向性证据，保留反向非确定性和正式梯度清理计数缺失等限制。

“唯一变化”必须由正式运行产物核验，不能只依据当前工作树或状态文档。每组固定检查：

- 训练入口及完整 argv；
- 数据集、七变量整窗语义、DataLoader batch/shuffle/drop_last；
- seed、公共初始化、A0/A1投影初始化和禁止warm-start；
- Adam、weight decay、nominal/effective LR与lradj、patience=3；
- bf16 Mamba、FiLM局部FP32；
- 每轮Test、最低raw-Val选轮及四horizon等权平均。

来源依次以正式目录的`source_snapshot/`、`manifest.json`、`command.json`、初始化/entry审计和结构化结果为准；当前`models/TimeMamba.py`及状态文档只作导航和辅助说明。若当前代码与正式快照不一致，必须按正式快照描述实验并明确差异。

## 3. 文档结构

### 3.1 顶部固定区域

1. 文档用途和扩展规则。
2. 当前统一实验口径。
3. P0/A0/A1总览表：实验状态、唯一变化、Avg MSE、Avg MAE、相对直接基准的变化和结论。
4. 模块索引，链接到各实验段落。

### 3.2 每个模块的统一模板

每个模块段落固定包含：

1. 实验编号、模块名称与验证问题。
2. 直接基准及唯一变化。
3. 保持不变的配置，避免把训练协议变化误算为模块贡献。
4. 四个 horizon 的“修改前 → 修改后”表格，分别列出基准选中轮和修改后选中轮、基准值、修改后值、MSE/MAE差值。
5. Avg MSE/MAE及相对变化。
6. 结果解释：正向、负向、混合结果及不能声称的事情。
7. 正式训练日志、`comparison.json`、逐项`entry_result.json`和必要审计文件的相对路径。

### 3.3 当前两个模块的直接比较

- A0：直接基准为P0；唯一变化是固定`H[:,:,:64]`替换为`Linear(768,64)`。
- A1：直接基准为A0；唯一变化是在同一投影前加入`Theta=alpha*H+(1-alpha)*E`，`alpha:[768]`初始0.5、无sigmoid/clamp。
- A1段落额外给出相对P0的整体结果，但不把A0投影和A1 residual的贡献混为一项。

所有差值统一定义为：

```text
delta = 修改后 - 直接基准
relative_change = delta / 直接基准 * 100%
```

误差指标中负delta表示改善，正delta表示退化。计算使用源文件完整精度，表格统一展示6位小数，百分比展示2位；平均值先用完整精度计算，再舍入展示，不能先舍入逐项再求平均。

## 4. 来源

- P0：`logs/ETTh1/ETTh1_96__2026-08-31-09-55.log`及A0正式目录的`baseline.json`/基准指纹。
- A0：`results/a0/formal_20260831T1710/comparison.json`、`baseline.json`、`source_snapshot/`，各horizon的`manifest.json`、`command.json`、`entry_result.json`、`completion.json`、`train.log`；初始化隔离证据来自`preflight/h*_p0/probe.json`和`preflight/h*_a0/probe.json`。
- A1：`results/a1/formal_20260831T2341/comparison.json`、`reference_a0.json`、`source_snapshot/`，各horizon的`manifest.json`、`command.json`、`initialization.json`、`entry_audit.json`、`entry_result.json`、`alpha_epochs.jsonl`、`completion.json`、`train.log`；A0/A1单步隔离证据来自`preflight/`。
- 模块定义优先使用上述正式`source_snapshot/models/TimeMamba.py`与manifest哈希；当前`models/TimeMamba.py`、已批准路线A规格及A0/A1状态文档只作辅助。

写入前必须从结构化汇总与原始日志复核数字和选轮，不能只转抄状态文档。

## 5. 扩展模板

文末提供一个不含虚构结果的占位模板。新增实验时：

- 使用新的稳定实验编号；
- 明确直接基准和唯一变化；
- 复制统一字段并填入真实路径；
- 同时更新顶部总览和模块索引；
- 未完成实验明确标为未完成，不预填指标或结论。

## 6. 验收

- A0、A1都具备完整的改动说明、逐horizon前后指标表、平均指标变化和日志路径。
- 所有平均值与差值可由源文件复算。
- A0/A1所有必需正式文件存在；各horizon状态为completed、来源未变化，manifest引用的基准/源码哈希一致。
- “唯一变化”表必须逐项核对入口、数据/loader、初始化、优化器/LR、精度、早停和选轮口径；不能只复述实验名称。
- 所有本地Markdown链接使用有效相对路径。
- 正负结果、单seed和选择协议限制表述准确。
- P0、A0、A1正式训练均没有逐步梯度清理计数。对应段落必须写明：不能用smoke的零计数、日志未见NaN或`nonfinite_reason=null`推断正式训练全程零清理；只允许说没有记录到显式中止或非有限评估指标。
- 文档可以在不改已有模块段落结构的情况下追加新模块。
- 不修改生产代码、实验日志或结果文件，不自动提交最终总结文档。
