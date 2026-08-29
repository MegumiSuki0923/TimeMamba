#!/bin/bash
set -eo pipefail

# 日志配置
data="Weather"
seq_len=96
comment="main_experiment"
timestamp=$(date +%Y-%m-%d-%H-%M)
log_file="./logs/${data}/${data}_${seq_len}_${comment}_${timestamp}.log"

# 确保日志目录存在
mkdir -p "./logs/${data}"

echo "=== 开始训练会话 ===" | tee -a "$log_file"
echo "日志文件: $log_file" | tee -a "$log_file"
echo "开始时间: $(date)" | tee -a "$log_file"

model_name=TimeMamba
train_epochs=100
learning_rate=0.001

# --- 硬件与加速配置 ---
master_port=00099
num_process=1
batch_size=32       # 保持大批次以加快训练速度

# 多尺度 patch 参数
small_patch=12
small_stride=4
large_patch=24
large_stride=12

d_model=32
d_ff=128
llm_layers=24

# Weather 数据集：21 个变量
enc_in=21
dec_in=21
c_out=21

comment="main_experiment"

echo "" | tee -a "$log_file"
echo "========================================" | tee -a "$log_file"
echo "# =========================================================
# 任务 1: 预测长度 96
# =========================================================" | tee -a "$log_file"
echo "========================================" | tee -a "$log_file"

# =========================================================
# 任务 1: 预测长度 96
# =========================================================
accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model $model_name \
  --data Weather \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --factor 3 \
  --enc_in $enc_in \
  --dec_in $dec_in \
  --c_out $c_out \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --lradj 'COS' \
  --dropout 0.2 \
  --llm_layers $llm_layers \
  --small_patch $small_patch \
  --small_stride $small_stride \
  --large_patch $large_patch \
  --large_stride $large_stride \
  --model_comment $comment \
  2>&1 | tee -a "$log_file"

echo "" | tee -a "$log_file"
echo "========================================" | tee -a "$log_file"
echo "# =========================================================
# 任务 2: 预测长度 192
# =========================================================" | tee -a "$log_file"
echo "========================================" | tee -a "$log_file"

# =========================================================
# 任务 2: 预测长度 192
# =========================================================
accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model $model_name \
  --data Weather \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 192 \
  --factor 3 \
  --enc_in $enc_in \
  --dec_in $dec_in \
  --c_out $c_out \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --lradj 'COS' \
  --dropout 0.2 \
  --llm_layers $llm_layers \
  --small_patch $small_patch \
  --small_stride $small_stride \
  --large_patch $large_patch \
  --large_stride $large_stride \
  --model_comment $comment \
  2>&1 | tee -a "$log_file"

echo "" | tee -a "$log_file"
echo "========================================" | tee -a "$log_file"
echo "# =========================================================
# 任务 3: 预测长度 336
# =========================================================" | tee -a "$log_file"
echo "========================================" | tee -a "$log_file"

# =========================================================
# 任务 3: 预测长度 336
# =========================================================
accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model $model_name \
  --data Weather \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 336 \
  --factor 3 \
  --enc_in $enc_in \
  --dec_in $dec_in \
  --c_out $c_out \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --lradj 'COS' \
  --dropout 0.2 \
  --llm_layers $llm_layers \
  --small_patch $small_patch \
  --small_stride $small_stride \
  --large_patch $large_patch \
  --large_stride $large_stride \
  --model_comment $comment \
  2>&1 | tee -a "$log_file"

echo "" | tee -a "$log_file"
echo "========================================" | tee -a "$log_file"
echo "# =========================================================
# 任务 4: 预测长度 720
# =========================================================" | tee -a "$log_file"
echo "========================================" | tee -a "$log_file"

# =========================================================
# 任务 4: 预测长度 720
# =========================================================
accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model $model_name \
  --data Weather \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 720 \
  --factor 3 \
  --enc_in $enc_in \
  --dec_in $dec_in \
  --c_out $c_out \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --lradj 'COS' \
  --dropout 0.2 \
  --llm_layers $llm_layers \
  --small_patch $small_patch \
  --small_stride $small_stride \
  --large_patch $large_patch \
  --large_stride $large_stride \
  --model_comment $comment \
  2>&1 | tee -a "$log_file"

echo "" | tee -a "$log_file"
echo "=== 训练会话结束 ===" | tee -a "$log_file"
echo "结束时间: $(date)" | tee -a "$log_file"
