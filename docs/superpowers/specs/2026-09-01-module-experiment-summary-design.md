# TimeMamba 模块实验总结文档设计

日期：2026-09-01
目标文件：`docs/module_experiment_summary.md`

## 1. 目标

建立一份面向后续模块筛选的统一总结文档，先完整收录 P0、A0、A1，未来可以用同一结构追加 A2、B0 等实验。文档必须回答：实验改了什么、修改前后四个 horizon 与平均指标如何变化、实验脚本和正式日志在哪里。

## 2. 证据核验规则（不作为最终文档独立章节）

最终总结文档删除独立的“数据与口径”章节，但写入前仍必须按以下规则核验。

“唯一变化”必须由正式运行产物核验，不能只依据当前工作树或状态文档。内部固定检查：

- 训练入口及完整 argv；
- 数据集、七变量整窗语义、DataLoader batch/shuffle/drop_last；
- seed、公共初始化、A0/A1投影初始化和禁止warm-start；
- Adam、weight decay、nominal/effective LR与lradj、patience=3；
- bf16 Mamba、FiLM局部FP32；
- 每轮Test、最低raw-Val选轮及四horizon等权平均。

来源依次以正式目录的`source_snapshot/`、`manifest.json`、`command.json`、初始化/entry审计和结构化结果为准；当前`models/TimeMamba.py`及状态文档只作导航和辅助说明。若当前代码与正式快照不一致，必须按正式快照描述实验并明确差异。

## 3. 文档结构

### 3.1 顶部

只保留文档标题和一句扩展说明，不建立独立的数据口径、总览或方法章节。

### 3.2 三个实验段落

P0、A0、A1均使用相同、可复制的三项结构：

1. **实验脚本**：给出可点击的脚本相对路径。
2. **Best Epoch、MSE、MAE表格**：
   - P0固定列为`Horizon｜Best Epoch｜MSE｜MAE`。
   - A0/A1固定列为`Horizon｜基准 Best Epoch｜基准 MSE｜基准 MAE｜实验 Best Epoch｜实验 MSE｜实验 MAE`。A0的基准为P0，A1的基准为A0。
   - 各表末尾增加`Avg`行；所有Best Epoch单元格写`—`，MSE/MAE使用四horizon完整精度算术平均后展示。
3. **日志文件**：列出正式训练日志、汇总结果和最关键审计文件的可点击相对路径。

每个实验标题后只增加一句事实性模块说明：

- P0：已完成的修正后基准。它包含七变量整窗/轴修正、FiLM局部FP32和普通Val早停，不伪装成论文模块替换，也不称为严格单因素实验。
- A0：控制实验；在P0上将固定`H[:,:,:64]`改为`Linear(768,64)`，没有直接论文模块归因。
- A1：根据T3Time，在A0上加入`Theta=alpha*H+(1-alpha)*E` residual fusion；`alpha:[768]`初始0.5，无sigmoid/clamp。

表格不增加delta或百分比列；修改前后数值直接并排。数值统一展示6位小数；平均值先用源文件完整精度计算，再舍入展示，不能先舍入逐项再求平均。

## 4. 来源

- P0：`logs/ETTh1/ETTh1_96__2026-08-31-09-55.log`及A0正式目录的`baseline.json`/基准指纹。
- A0：`results/a0/formal_20260831T1710/comparison.json`、`baseline.json`、`source_snapshot/`，各horizon的`manifest.json`、`command.json`、`entry_result.json`、`completion.json`、`train.log`；初始化隔离证据来自`preflight/h*_p0/probe.json`和`preflight/h*_a0/probe.json`。
- A1：`results/a1/formal_20260831T2341/comparison.json`、`reference_a0.json`、`source_snapshot/`，各horizon的`manifest.json`、`command.json`、`initialization.json`、`entry_audit.json`、`entry_result.json`、`alpha_epochs.jsonl`、`completion.json`、`train.log`；A0/A1单步隔离证据来自`preflight/`。
- 模块定义优先使用上述正式`source_snapshot/models/TimeMamba.py`与manifest哈希；当前`models/TimeMamba.py`、已批准路线A规格及A0/A1状态文档只作辅助。
- 三个实验脚本固定为：P0=`scripts/ETTh1.sh`、A0=`scripts/ETTh1_a0.sh`、A1=`scripts/ETTh1_a1.sh`。若当前脚本哈希与正式快照不一致，最终文档改为链接对应正式目录下的`source_snapshot/scripts/`版本并标明是运行快照；一致时链接当前脚本。

写入前必须从结构化汇总与原始日志复核数字和选轮，不能只转抄状态文档。

## 5. 扩展模板

文末提供一个不含虚构结果的占位模板，结构严格为“实验脚本、指标表、日志文件”。新增实验时：

- 使用新的稳定实验编号；
- 明确直接基准和唯一变化；
- 复制统一字段并填入真实路径；
- 未完成实验明确标为未完成，不预填指标或结论。

## 6. 验收

- P0、A0、A1都具备一句模块说明、实验脚本、逐horizon指标表和日志路径。
- P0不得套用“根据论文替换模块”的表述；A0明确为控制实验；只有A1标注T3Time来源。
- 所有平均值与差值可由源文件复算。
- A0/A1所有必需正式文件存在；各horizon状态为completed、来源未变化，manifest引用的基准/源码哈希一致。
- 内部必须逐项核对入口、数据/loader、初始化、优化器/LR、精度、早停和选轮口径；不把这些核验过程扩展成最终文档独立章节。
- 所有本地Markdown链接使用有效相对路径。
- 表格数字保持原始单seed证据，不增加统计显著、稳定收益或零异常梯度等源文件不支持的结论。
- 文档可以在不改已有模块段落结构的情况下追加新模块。
- 不修改生产代码、实验日志或结果文件，不自动提交最终总结文档。
