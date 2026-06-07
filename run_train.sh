#!/bin/bash

conda activate team15

mkdir -p log

declare -l TRAIN_VALUE

TRAIN_VALUE=$1

if [ "$TRAIN_VALUE" -eq "true" ]; then
    python3 main.py \
        --seed 42 \
        --batch 8 \
        --epochs 30 \
        --warmup_epochs 2 \
        --embed_dim 256 \
        --num_heads 8 \
        --mlp_ratio 2.0 \
        --drop 0.1 \
        --attn_drop 0.1 \
        --lr 3e-4 \
        --backbone_lr_mult 0.1 \
        --weight_decay 0.05 \
        --weight_pow 0.5 \
        --max_class_weight 10.0 \
        --workers 8 \
        --dataset both \
        > log/log_v3_42_both.log \
        2> log/err_v3_42_both.log

else
    python3 main.py \
        --seed 42 \
        --train ${TRAIN_VALUE} \
        > log/log_v3_42_both.log \
        2> log/err_v3_42_both.log