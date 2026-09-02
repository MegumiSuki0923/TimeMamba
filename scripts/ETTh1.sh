#!/bin/bash
# B4-derived minimal baseline: seed 2025, four horizons, serial execution.
set -eo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

data="ETTh1"
seq_len=96
seed=2025
accelerate_bin=/home/Lain/anaconda3/envs/time-llm/bin/accelerate
train_epochs=100
weight_decay=0.01
batch_size=24
small_patch=8
small_stride=4
large_patch=24
large_stride=4
d_model=32
d_ff=64
llm_layers=24
master_port=29506

run_id="formal_$(date +%Y%m%dT%H%M%S)"
result_root="./results/baseline/${run_id}"
log_file="./logs/ETTh1/baseline_${run_id}.log"
mkdir -p "./logs/ETTh1"
exec > >(tee -a "$log_file") 2>&1

echo "=== TimeMamba minimal baseline ${run_id} ==="
echo "开始时间: $(date)"

run_task() {
  local pred=$1 lr=$2 lradj=$3 outdir=$4
  mkdir -p "$outdir"
  echo "----------------------------------------"
  echo "开始 run_main.py seed=${seed} pred_len=${pred}"
  $accelerate_bin launch --mixed_precision bf16 --num_processes 1 \
    --main_process_port $master_port run_main.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model TimeMamba \
    --data $data \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des Exp \
    --itr 1 \
    --d_model $d_model \
    --d_ff $d_ff \
    --batch_size $batch_size \
    --learning_rate $lr \
    --weight_decay $weight_decay \
    --train_epochs $train_epochs \
    --dropout 0.2 \
    --lradj $lradj \
    --clip_grad 0.5 \
    --llm_layers $llm_layers \
    --small_patch $small_patch \
    --small_stride $small_stride \
    --large_patch $large_patch \
    --large_stride $large_stride \
    --seed $seed \
    --model_comment baseline \
    --result_json "${outdir}/entry_result.json"
}

run_task 96  0.01  type1 "${result_root}/h96"
run_task 192 0.02  type1 "${result_root}/h192"
run_task 336 0.001 COS   "${result_root}/h336"
run_task 720 0.01  type1 "${result_root}/h720"

echo "=== TimeMamba minimal baseline finished ==="
echo "结束时间: $(date)"
