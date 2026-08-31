#!/bin/bash
set -eo pipefail

# 日志配置
data="ETTh1"
seq_len=96
comment="main_experiment"
timestamp=$(date +%Y-%m-%d-%H-%M)
log_file="./logs/${data}/${data}_${seq_len}_${comment}_${timestamp}.log"

# 确保日志目录存在
mkdir -p "./logs/${data}"

exec > >(tee -a "$log_file") 2>&1
export PYTHONUNBUFFERED=1

echo "=== 开始训练会话 ==="
echo "日志文件: $log_file"
echo "开始时间: $(date)"

model_name=TimeMamba
accelerate_bin=/home/Lain/anaconda3/envs/time-llm/bin/accelerate
train_epochs=100
learning_rate=0.0001
weight_decay=0.01

# --- 硬件与加速配置 ---
master_port=29500
num_process=1
batch_size=24

# 多尺度 patch 参数
small_patch=8
small_stride=4
large_patch=24
large_stride=4

d_model=32
d_ff=64
llm_layers=24

# =========================================================
# 任务 1: 预测长度 96
# =========================================================
echo ""
echo "========================================"
echo "开始 pred_len=96"
echo "========================================"

$accelerate_bin launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model $model_name \
  --data $data \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 96 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --weight_decay $weight_decay \
  --train_epochs $train_epochs \
  --dropout 0.2 \
  --lradj COS \
  --clip_grad 0.5 \
  --llm_layers $llm_layers \
  --small_patch $small_patch \
  --small_stride $small_stride \
  --large_patch $large_patch \
  --large_stride $large_stride \
  --model_comment "$comment"

# =========================================================
# 任务 2: 预测长度 192
# =========================================================
echo ""
echo "========================================"
echo "开始 pred_len=192"
echo "========================================"

$accelerate_bin launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model $model_name \
  --data $data \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 192 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --weight_decay $weight_decay \
  --train_epochs $train_epochs \
  --dropout 0.2 \
  --lradj COS \
  --clip_grad 0.5 \
  --llm_layers $llm_layers \
  --small_patch $small_patch \
  --small_stride $small_stride \
  --large_patch $large_patch \
  --large_stride $large_stride \
  --model_comment "$comment"

# =========================================================
# 任务 3: 预测长度 336
# =========================================================
echo ""
echo "========================================"
echo "开始 pred_len=336"
echo "========================================"

$accelerate_bin launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model $model_name \
  --data $data \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 336 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --weight_decay $weight_decay \
  --train_epochs $train_epochs \
  --dropout 0.2 \
  --lradj COS \
  --clip_grad 0.5 \
  --llm_layers $llm_layers \
  --small_patch $small_patch \
  --small_stride $small_stride \
  --large_patch $large_patch \
  --large_stride $large_stride \
  --model_comment "$comment"

# =========================================================
# 任务 4: 预测长度 720
# =========================================================
echo ""
echo "========================================"
echo "开始 pred_len=720"
echo "========================================"

$accelerate_bin launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model $model_name \
  --data $data \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 720 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --weight_decay $weight_decay \
  --train_epochs $train_epochs \
  --dropout 0.2 \
  --lradj COS \
  --clip_grad 0.5 \
  --llm_layers $llm_layers \
  --small_patch $small_patch \
  --small_stride $small_stride \
  --large_patch $large_patch \
  --large_stride $large_stride \
  --model_comment "$comment"

echo ""
echo "========================================"
echo "自动清理日志并提取结果"
echo "========================================"
python utils/clean_log.py "$log_file"

echo ""
echo "=== 训练会话结束 ==="
echo "结束时间: $(date)"
