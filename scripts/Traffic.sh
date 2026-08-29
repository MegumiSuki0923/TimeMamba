#!/usr/bin/env bash
set -eo pipefail

# 日志配置
data="Traffic"
seq_len=512
comment="sp12-st6-lp24-ls12"
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
master_port=00102
num_process=1
batch_size=128

# 多尺度 patch 参数（Traffic 小时级数据，seq_len=512）
# small: 12h 半天模式（早/晚高峰）→ (512-12)/6+1 = 84 tokens
# large: 24h 全天日周期       → (512-24)/12+1 = 41 tokens
# 注意：large_patch 必须 < seq_len=512，否则无法切出 token
small_patch=12
small_stride=6
large_patch=24
large_stride=12

d_model=32
d_ff=128
llm_layers=24

# Traffic 数据集：862 个变量（除 date 外共 862 列）
enc_in=862
dec_in=862
c_out=862

comment="sp12-st6-lp24-ls12"

run_traffic() {
  local pred_len="$1"

  accelerate launch --mixed_precision bf16 --num_processes "$num_process" --main_process_port "$master_port" run_main.py \
    --is_training 1 \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --model "$model_name" \
    --data Traffic \
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
    --num_workers 4 \
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
# Traffic: 预测长度 96 / 192 / 336 / 720
# =========================================================
run_traffic 96
run_traffic 192
run_traffic 336
run_traffic 720

echo "" | tee -a "$log_file"
echo "=== 训练会话结束 ===" | tee -a "$log_file"
echo "结束时间: $(date)" | tee -a "$log_file"
