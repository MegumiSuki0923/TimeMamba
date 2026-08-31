# TimeMamba T3Time 多变量协议与 Residual Fusion 设计

日期：2026-08-31  
状态：设计段落已由 Lain 批准；规格审查通过，全文待 Lain 审阅
范围：ETTh1 第一轮单模块筛选；不是最终论文架构

## 1. 目标

先把当前 TimeMamba 从 channel-independent 数据样本协议改为与 T3Time 对齐的真正多变量协议，再依次运行 P0、A0、A1 三组控制实验，判断：

1. TimeMamba 在 T3Time 数据与模型选择协议下的真实起点；
2. 固定截取 Mamba 隐状态前 `d_ff` 维是否构成信息瓶颈；
3. T3Time 官方代码式 residual fusion 是否能在冻结 Mamba 场景中独立提升预测。

筛选阶段使用单 seed。四个预测长度的算术平均同时达到 `MSE <= 0.418`、`MAE <= 0.430` 后，P0、A0、A1 使用同一组 3 seeds 复验。路线 A 未达到目标时不继续叠加模块，保留结果并转入后续候选路线。

本设计只覆盖路线 A。最终论文还会探索其他 2025–2026 正式顶会模块；预训练归因、alpha 深度分析和完整论文级归因等实验延后到最终结构确定后，不阻塞本阶段。

## 2. 非目标

- 不在本阶段加入跨变量 mixer、TimeFilter、xCPD 或其他 channel-dependency 模块。
- 不替换 MSTF、层次统计条件 tokens、冻结 Mamba、FlattenHead 或 MSE loss。
- 不为旧的 channel-independent Dataset 行为编写向后兼容分支。
- 不在路线 A 上继续堆叠 decoder、frequency encoder 或新 loss。
- 不把 T3Time residual fusion 声称为原创模块。
- 不在本阶段完成正式论文的全部归因与多数据集实验。

## 3. 官方依据与实现边界

T3Time 正式论文：

- https://ojs.aaai.org/index.php/AAAI/article/view/39196

T3Time 官方代码：

- 训练与模型选择：https://github.com/monaf-chowdhury/T3Time/blob/main/train.py
- 模型与 residual：https://github.com/monaf-chowdhury/T3Time/blob/main/models/T3Time.py
- ETTh1 配置：https://github.com/monaf-chowdhury/T3Time/blob/main/scripts/ETTh1.sh

论文将 residual coefficient 描述为 `[0,1]` 内的凸组合，但官方代码仅把 `residual_alpha` 初始化为 `0.5`，没有 sigmoid 或 clamp。Lain 已决定采用官方代码版：A1 使用无约束可训练 `alpha`，初始值为 `0.5`。

## 4. 当前模块与职责

当前 TimeMamba 的非骨干路径为：

1. `StandardScaler`：用训练集统计量缩放各变量。
2. `Normalize`/RevIN：在每个输入窗口内做实例归一化，预测后恢复窗口尺度。
3. `HierarchicalDynamicPrompt`：从每个变量的 12 维统计量生成 meta、pattern、numerical 三组连续条件 tokens。
4. `MultiScaleTimeFreqFusion`：用小 patch、大 patch和全局 FFT 幅值构造 `[B*N,P,768]` 数值 tokens。
5. 冻结 Mamba：处理条件 tokens 与数值 tokens。
6. 固定隐藏维切片：当前只取 Mamba 输出的前 `d_ff` 个隐藏坐标。
7. `FlattenHead`：展平 patch 与隐藏维并一次性输出 `pred_len` 个时间点。

路线 A 只修改第 6 项及其前后的信息路径。

## 5. 多变量协议 P0

### 5.1 Dataset 语义

ETTh1 的每个 Dataset item 必须返回一个完整七变量窗口：

- `seq_x: [96,7]`
- `seq_y: [pred_len,7]`
- 时间标记保持现有维度

Dataset 长度为时间窗口数量，不再乘以变量数。删除 `feat_id` 索引语义。训练、验证、测试使用与 T3Time 一致的数据边界和训练集 StandardScaler。

`label_len=0`。当前 TimeMamba 不消费 decoder 输入，因此不保留无效的历史 decoder 段。

### 5.2 DataLoader 语义

train、validation、test 均设置：

- `shuffle=False`
- `drop_last=True`

四个 horizon 的逻辑 batch size 与 T3Time ETTh1 官方脚本一致：

| pred_len | logical batch size | epochs |
|---:|---:|---:|
| 96 | 256 | 150 |
| 192 | 32 | 150 |
| 336 | 16 | 120 |
| 720 | 32 | 150 |

逻辑 batch 决定 DataLoader 长度、丢弃的尾部样本和一次 optimizer step 的有效样本集合。

### 5.3 Micro-batch

若逻辑 batch 无法一次通过冻结 Mamba，则允许配置 micro-batch：

- DataLoader 仍产生完整逻辑 batch；
- 训练时将逻辑 batch切成若干 micro-batch；
- 每个 micro-batch 的 loss 按其元素数占逻辑 batch 总元素数的比例加权；
- 所有 micro-batch 累积完梯度后只执行一次 gradient clip 和 optimizer step；
- validation/test 也可以分块前向，但指标必须按元素数正确聚合；
- 不得改变 `drop_last` 后保留的样本集合。

micro-batch 是资源执行方式，不是新的模型变体。同一 horizon 的 P0/A0/A1 使用相同 micro-batch 大小，并在筛选前通过三组 smoke test 确定和记录。它保留逻辑 batch、样本集合和 optimizer step 语义，但开启 dropout 时随机掩码及浮点运算顺序可能不同，不能声称与整批训练轨迹逐位等价。

### 5.4 N>1 张量修正

当前 `forecast` 中把 `[B,T,N]` 直接 `reshape(B,N,T)` 后再 permute 的路径会在 `N>1` 时打乱元素。P0 必须删除该错误 reshape，使用显式 `permute` 保持时间与变量轴语义。

P0 虽然接收完整 `[B,T,7]`，MSTF 之后仍把变量展平为 `B*N` 独立处理。P0 是真正多变量的数据/输出协议，但不是 channel-mixing 架构，不得宣称已经建模跨变量依赖。

## 6. P0、A0、A1 模型定义

### 6.1 P0：协议 baseline

P0 只包含多变量协议和必要的 N>1 张量修正。模型输出路径保持：

```text
MSTF E:[B*N,P,768]
  -> frozen Mamba H:[B*N,P,768]
  -> H[:,:,:d_ff]
  -> FlattenHead
  -> [B,pred_len,N]
```

P0 用于建立新协议下的 TimeMamba 起点。P0 与旧 CI 日志的差异不能归因给任何新模型模块。

### 6.2 A0：学习投影控制组

A0 在 P0 上只改变隐藏维选择：

```text
固定 H[:,:,:d_ff]
  -> Linear(768,d_ff)
```

只对最后 `P` 个数值 patch 对应的 Mamba 输出执行投影。A0 不加入 residual。`A0-P0` 只衡量可学习隐藏维选择相对固定坐标切片的影响。

### 6.3 A1：T3Time 官方代码式 residual

A1 与 A0 使用同构、相同初始权重的 learned projection，但各自独立训练，不共用训练后的参数，也不从 A0 checkpoint 初始化 A1。新增的唯一模块为：

```text
E = MSTF 原始数值 patch tokens      [B*N,P,768]
H = Mamba 最后 P 个数值 token 输出  [B*N,P,768]
alpha                               [768], init=0.5
Theta = alpha * H + (1-alpha) * E   [B*N,P,768]
Z = Linear(768,d_ff)(Theta)         [B*N,P,d_ff]
```

`alpha` 按隐藏维广播到 batch 和 patch 轴。它不经过 sigmoid，也不 clamp。训练日志必须记录 alpha 的最小值、最大值、均值和标准差，以确认实际行为；这些统计仅作为实现诊断，本阶段不要求完整机制分析。

`A1-A0` 是 residual 的独立贡献，`A1-P0` 是整条信息保真路线的总贡献。

### 6.4 变体接口

实现应提供一个显式、互斥的实验变体参数，例如 `p0 | a0 | a1`。非法值立即报错。不要通过隐式参数组合推断变体，也不要保留旧行为的兼容性别名。

## 7. 训练与模型选择协议

所有变体固定：

- screening seed：2024
- AdamW，`learning_rate=1e-4`
- `weight_decay=1e-3`
- CosineAnnealingLR，`T_max=min(epochs,50)`，`eta_min=1e-6`
- gradient clip：5
- early-stopping patience：25
- 保留当前 TimeMamba 的 bf16 执行方式；同一 horizon 的 P0/A0/A1 使用相同精度和单 GPU 执行配置，不宣称完全复制 T3Time 的数值计算轨迹

“保留当前模型配置”具体指 `scripts/ETTh1_seq96.sh` 的模型参数，而不是 CLI 默认值：

| 配置 | 固定值 |
|---|---:|
| d_model | 32 |
| d_ff | 64 |
| small_patch / small_stride | 8 / 4 |
| large_patch / large_stride | 24 / 4 |
| llm_layers | 24 |
| dropout | 0.2 |
| prompt_tokens | 16 |
| num_pattern_types | 8 |
| seq_len / label_len / enc_in | 96 / 0 / 7 |
| percent / train_stride | 100 / 1 |

其中 prompt 与 prototype 数量来自当前模型/入口默认值。本轮四个 horizon 均使用上表设置，只有 pred_len、逻辑 batch 和 epoch 上限按前表变化。

每个 horizon、每个 seed 内的初始化与训练隔离要求：

- P0/A0/A1 均从头独立训练，禁止跨变体 warm-start。
- 三者的公共模块使用相同初始参数和 buffers，包括同一份冻结 Mamba 预训练权重。
- A0/A1 的 projection 使用相同初始化，按标准 `nn.Linear` 初始化；A1 alpha 固定初始化为 0.5。
- 构造新增模块时隔离其随机数消耗，或显式复制公共初始 state；不能仅依赖“相同 seed”保证公共权重相同。
- 模型构造完成后重置训练 RNG；记录公共初始 state 和 projection 的校验摘要，并测试对应参数逐项相等。
- “同一 projection”指相同结构和初始化，不表示训练中绑权，也不表示沿用 A0 训练结果。

模型选择完全复刻 T3Time 官方代码语义：

1. 初始化 validation threshold 和 test threshold 为正无穷。
2. 前 10 个 epoch：validation MSE 低于当前 validation threshold 时保存 checkpoint，并更新 validation threshold。
3. 第 10 个 epoch 后：当 validation MSE 低于当前 threshold 时评估 test。
4. 仅当 test MSE 低于 test threshold 时保存 checkpoint，同时更新 test threshold 和 validation threshold。
5. 若 test 未改善，不更新 validation threshold；后续低于旧 threshold 的 epoch仍可再次触发 test。
6. `epochs_since_best_mse` 与官方代码一致；达到 patience 且训练已过总 epoch 的一半时早停。
7. 训练结束后恢复保存的 checkpoint，在同一 `drop_last=True` test loader 上报告 MSE/MAE。

这是 test-guided selection。日志、结果表和未来论文必须明确标注，不能描述为 validation-only selection。

## 8. 指标与成功标准

每个变体分别训练 `pred_len in {96,192,336,720}`。MSE 和 MAE 在所有保留的测试样本、所有预测步和全部七个变量上计算。

最终汇总：

```text
Avg MSE = (MSE_96 + MSE_192 + MSE_336 + MSE_720) / 4
Avg MAE = (MAE_96 + MAE_192 + MAE_336 + MAE_720) / 4
```

路线 A 单 seed 成功条件：A1 同时满足 `Avg MSE <= 0.418` 和 `Avg MAE <= 0.430`。

成功后，P0、A0、A1 都使用同一组 3 seeds 重跑，报告 mean 和 standard deviation。不能只对 A1 做多 seed。

## 9. 验证要求

### 9.1 数据与形状

- Dataset item 的 `x/y` 末维为 7。
- Dataset 长度等于窗口数而不是窗口数乘 7。
- DataLoader 的逻辑 batch、loader 长度和 `drop_last` 样本数与配置一致。
- P0/A0/A1 对 `B>=2,N=7` 均输出 `[B,pred_len,7]`。
- 显式检查 N>1 时变量轴没有被 reshape 打乱。

### 9.2 变体隔离

- P0 不实例化 learned projection 或 alpha。
- A0 实例化 projection，不实例化 alpha。
- A1 实例化同构且与 A0 相同初始化的 projection 与 alpha。
- 各组公共参数及 buffers 在训练前逐项一致，禁止跨变体加载已训练 checkpoint。
- A0/P0 的参数差只来自 projection；A1/A0 的参数差只来自 alpha。
- projection 和 alpha 均获得有限非零梯度。

### 9.3 Micro-batch 等价性

在 dropout 关闭的受控小 batch 上，比较不分块和分块的：

- loss；
- 参数梯度；
- optimizer step 后参数。

使用合理浮点容差。评估分块与不分块的 MSE/MAE 必须一致。

### 9.4 模型选择

用合成的 validation/test loss 序列覆盖：

- epoch 1–10 的 validation 保存；
- epoch 10 后 validation 触发 test；
- test 不改善时不更新 validation threshold；
- test 改善时更新 checkpoint；
- 早停条件；
- 最终恢复的是正确 checkpoint。

## 10. 失败处理与停止条件

- Gate 0 失败：只修复数据、形状或协议，不进入 P0 长跑。
- P0 出现 non-finite、OOM 或错误样本数：修复协议执行问题后重跑；不得直接调模型模块。
- A0 未改善：仍允许按计划测试 A1，因为 residual 可能独立有效；保留 A0 负结果。
- A1 未优于 A0：记录为“当前单 seed、配置和协议下未观察到 residual 的独立收益”，不推广成普遍无效，也不把它写成已验证有效模块。MSE/MAE 一升一降时分别报告，不强行判定统一改善。
- A1 有改善但未达到目标：结束路线 A，不继续在 A1 上堆模块；转入已批准的下一条独立模块路线。
- A1 达标：进入 P0/A0/A1 三变体 3-seed 复验，再决定是否纳入最终架构。

## 11. 论文表述边界

允许采用“领域问题 -> 设计动机 -> 方法 -> 验证”的叙事，而不是描述工程试错过程。建议的领域问题是：冻结预训练序列骨干可能削弱任务原生数值证据，单一路径和固定隐藏维选择进一步造成表示损失。

必须遵守：

- 明确引用 T3Time，并说明 residual 机制来自其官方实现。
- 不把 residual fusion 改名后声称原创。
- 只有 A1 稳定优于 A0 时，才能声称原始 MSTF 数值旁路在冻结 Mamba 中有效。
- 达到目标只是进入论文候选阶段的性能门槛，不自动证明创新性。
- 可主张的潜在贡献是对冻结 SSM 数值信息路径的诊断、接口适配和受控验证；最终贡献仍取决于后续其他模块和完整实验。

预训练归因、alpha 深度机制分析和完整论文级非模块实验延后到最终网络结构确定后统一设计，不属于本阶段实施阻塞项。

## 12. 交付边界

路线 A 的实现计划应覆盖：

- ETTh1 多变量 Dataset/DataLoader；
- N>1 张量修正；
- P0/A0/A1 模型变体；
- T3Time 训练、checkpoint 和 test-guided selection；
- logical batch 与 micro-batch；
- 单元测试、shape smoke test 和单步训练 smoke test；
- 四 horizon 串行运行脚本与结构化结果汇总。

实现计划不得自动启动四个正式长跑，除非 Lain 在代码和 smoke test 审阅后另行批准。

## 13. 规格审查记录

2026-08-31 第一轮发现配置来源和跨变体初始化不明确；补齐第 5.3、6.3、7、9.2 和 10 节后，第二轮审查结果为 Approved，无剩余阻塞问题。模型代码和正式训练尚未开始。
