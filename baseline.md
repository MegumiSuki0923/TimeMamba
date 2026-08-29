# 基线
- 日期：2026-08-27
- 状态：完成
- 训练日志：explore_timellm-main/logs/ETTh1/ETTh1_512_base_2026-08-27_14:13.log
- 训练脚本：scripts/ETTh1.sh
- 实验方案：Time-LLM GPT2 原始代码，未做任何更改

- 参考来源：无
- 运行方式：bash scripts/ETTh1.sh
- 实验效果:

| pred_len | best MSE | best MAE(with best MSE) | best epoch | speed      |
| -------- | -------- | ----------------------- | ---------- | ---------- |
| 96       | 0.3902   | 0.4096                  | 3          | 31.18 it/s |
| 192      | 0.4354   | 0.4391                  | 2          | 31.18 it/s |
| 336      | 0.4745   | 0.4613                  | 2          | 31.00 it/s |
| 720      | 0.4588   | 0.4729                  | 4          | 31.00 it/s |
| Avg      | 0.4397   | 0.4457                  | -          | 31.09 it/s |
