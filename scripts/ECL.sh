#!/usr/bin/env bash
set -eo pipefail

# 日志配置
data="ECL"
seq_len=512
comment="主实验结果"
timestamp=$(date +%Y-%m-%d-%H-%M)
log_file="./logs/${data}/${data}_${seq_len}_${comment}_${timestamp}.log"

# 确保日志目录存在
mkdir -p "./logs/${data}"

echo "=== 开始训练会话 ===" | tee -a "$log_file"
echo "日志文件: $log_file" | tee -a "$log_file"
echo "开始时间: $(date)" | tee -a "$log_file"


set -euo pipefail

model_name=TimeMamba
train_epochs=100
learning_rate=1e-4

# --- 硬件与加速配置 ---
master_port=00101
num_process=1
batch_size=384

# 多尺度 patch 参数（ECL 小时级数据，seq_len=512）
# small: 16h 日内模式（早晚高峰）, large: 48h 跨日周期
small_patch=16
small_stride=8
large_patch=48
large_stride=8

d_model=32
d_ff=128
llm_layers=24

# ECL 数据集：321 个变量（除 date 外共 321 列）
enc_in=321
dec_in=321
c_out=321

comment="主实验结果"

run_ecl() {
  local pred_len="$1"

  accelerate launch --mixed_precision bf16 --num_processes "$num_process" --main_process_port "$master_port" run_main.py \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model "$model_name" \
    --data ECL \
    --features M \
    --freq h \
    --seq_len 512 \
    --label_len 48 \
    --pred_len "$pred_len" \
    --factor 3 \
    --enc_in "$enc_in" \
    --dec_in "$dec_in" \
    --c_out "$c_out" \
    --des 'Exp' \
    --itr 1 \
    --d_model "$d_model" \
    --d_ff "$d_ff" \
    --batch_size "$batch_size" \
    --learning_rate "$learning_rate" \
    --train_epochs "$train_epochs" \
    --dropout 0.2 \
    --llm_layers "$llm_layers" \
    --small_patch "$small_patch" \
    --small_stride "$small_stride" \
    --large_patch "$large_patch" \
    --large_stride "$large_stride" \
    --model_comment "$comment"
} \
  2>&1 | tee -a "$log_file"

# =========================================================
# ECL: 预测长度 96 / 192 / 336 / 720
# =========================================================
run_ecl 96
run_ecl 192
run_ecl 336
run_ecl 720

echo "" | tee -a "$log_file"
echo "=== 训练会话结束 ===" | tee -a "$log_file"
echo "结束时间: $(date)" | tee -a "$log_file"
