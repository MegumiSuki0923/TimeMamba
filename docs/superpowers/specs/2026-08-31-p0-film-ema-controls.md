# TimeMamba P0：FiLM 精度与早停补充对照

Lain 已批准实施两组单因素对照。环境为 conda time-llm，模型为 TimeMamba；不使用 TimeLLM 模型或 T3Time 训练协议。

## 参照与两个实验

参照是已完成的 `logs/ETTh1/ETTh1_96__2026-08-31-09-55.log`：七变量整窗 P0，FiLM 局部 FP32，原始 validation loss 早停。

| 组别 | FiLM meta 调制 | 停止判据 | 唯一变化 |
|---|---|---|---|
| 已完成 P0 | 局部 FP32 | raw validation，patience3 | 参照 |
| film_bf16 | 原始 autocast/bf16 算式 | raw validation，patience3 | 撤销精度修复 |
| ema | 局部 FP32 | validation EMA，alpha0.5，patience3 | 恢复平滑停止判据 |

每组跑 pred_len 96/192/336/720，共8项正式训练，单 GPU 串行，先 film_bf16 再 ema。各组各 horizon 独立从头训练，不加载别组 checkpoint，不自动运行 A0/A1。

## 固定配置

继承当前 `scripts/ETTh1.sh`：seq_len96、label_len48、features M、batch24、seed2025、100轮上限、Adam/weight_decay0.01、dropout0.2、clip_grad0.5、d_model32/d_ff64、24层冻结 Mamba、patch8/24和stride4、prompt16和prototype8。维持原 DataLoader、七变量整窗、轴修正、bf16 和每轮 val/test。

| horizon | 命令行 learning_rate | lradj |
|---|---:|---|
| 96 | 0.01 | type1 |
| 192 | 0.02 | type1 |
| 336 | 0.001 | COS |
| 720 | 0.01 | type1 |

原入口 OneCycle 初始化后 type1 的实际初始 LR 与命令行不同，必须保持现有行为，不进行调度器修正。

## 实现隔离

- 不修改正式 `run_main.py`、`utils/tools.py`、`scripts/ETTh1.sh` 或模型文件，也不将 EMA 参数加回生产入口。
- 独立实验包装器调用 `runpy.run_path(run_main.py)`，仍由原入口构建模型、优化器、DataLoader、损失及指标。
- film_bf16 只替换 HierarchicalDynamicPrompt 的 forward 为修复前算式；继承相同构造函数，保持参数名、shape、初始化和 RNG 一致。
- ema 只替换进程内 EarlyStopping 类，保留已移除实现的 alpha0.5、delta0、patience3，无最低轮数约束。
- 两组固定 save_checkpoint=False；EMA 仅改变停止时机，不改变原入口 best_metrics 按 raw validation 选轮的逻辑。
- 包装器退出后恢复进程内补丁，不产生旧参数的兼容别名。

## 梯度清理与审计

原入口已有 `torch.nan_to_num_(grad, nan=0, posinf=0, neginf=0)`。两组保留完全相同的清理行为，不能为了让一组通过而另加策略。进程内观测器记录清理前非有限元素和触发张量调用数量；只读计数，不改变数值、RNG 或 optimizer step。

记录完整 argv、环境、原文件 SHA256、模型初始 state 摘要、可训练参数量、数据形状和梯度清理计数。额外结果文件、日志标签不改变训练计算。源文件如与已验收 P0 不一致则拒绝启动，不静默混合版本。

## 指标与结论

两组均取每个 horizon 最低 **原始 validation loss** 轮的 Test MSE/MAE，再对四 horizon 算术平均。不能切换到最低 Test、EMA checkpoint 或从多个指标组合结果。

参照 P0 平均值：MSE 0.461188675，MAE 0.460331675。逐 horizon：

| horizon | selected epoch | MSE | MAE |
|---|---:|---:|---:|
| 96 | 4 | 0.3838765 | 0.4071353 |
| 192 | 8 | 0.4310462 | 0.4394976 |
| 336 | 2 | 0.5124965 | 0.4881754 |
| 720 | 11 | 0.5173355 | 0.5065184 |

对每个对照计算 `P0 - control`。仅当两个平均误差都不高于对照，才描述为“此单 seed 实验未观察到该项改动的负面效果”。混合变化、退化、崩溃、非有限指标均如实报告；不能把训练失败填为一个 MSE/MAE，也不能据单 seed 断言普遍无害。

缺失、失败、混合组别、混合 seed 或混合源码的结果不汇总成完整均值。一个组失败不应伪装完成；另一个独立组可继续执行。

## 验收

- 源文件 hash 在实现、测试和训练后保持不变。
- 同 seed 的模型 state 与参数量相同；关闭 autocast 时两个 prompt forward 相同；autocast 下只有目标 meta 路径精度不同。
- EMA 停止序列与旧公式一致；raw 模式保持现有实现；报告选轮不变。
- 梯度观测前后数值、梯度清理结果及 RNG 相同。
- CPU 回归和参数契约检查通过后启动八项正式训练；检查首轮日志确认实际模型、早停组别及环境。
