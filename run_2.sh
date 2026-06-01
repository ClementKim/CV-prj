#!/bin/bash

conda activate CV

python3 main.py \
    --seed 42 \
    --batch 16 \
    --epochs 30 \
    --patch_size 16 \
    --embed_dim 256 \
    --num_heads 8 \
    --mlp_ratio 2.0 \
    --drop 0.1 \
    --attn_drop 0.1 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    > log/log_v2_42_256.log \
    2> log/err_v2_42_256.log

python3 main.py \
    --seed 42 \
    --batch 16 \
    --epochs 30 \
    --patch_size 16 \
    --embed_dim 512 \
    --num_heads 8 \
    --mlp_ratio 2.0 \
    --drop 0.1 \
    --attn_drop 0.1 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    > log/log_v2_42_512.log \
    2> log/err_v2_42_512.log