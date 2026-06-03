#!/bin/bash
# epochs 20 -> 30으로 조정
conda activate CV

python3 main.py \
    --seed 42 \
    --batch 16 \
    --epochs 20 \
    --patch_size 16 \
    --embed_dim 256 \
    --num_heads 8 \
    --mlp_ratio 2.0 \
    --drop 0.1 \
    --attn_drop 0.1 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --dataset both \
    > log/log_v2_42_1e4.log \
    2> log/err_v2_42_1e4.log

python3 main.py \
    --seed 42 \
    --batch 16 \
    --epochs 20 \
    --patch_size 16 \
    --embed_dim 256 \
    --num_heads 8 \
    --mlp_ratio 2.0 \
    --drop 0.1 \
    --attn_drop 0.1 \
    --lr 2e-4 \
    --weight_decay 0.01 \
    --dataset both \
    > log/log_v2_42_2e4.log \
    2> log/err_v2_42_2e4.log

python3 main.py \
    --seed 42 \
    --batch 16 \
    --epochs 20 \
    --patch_size 16 \
    --embed_dim 256 \
    --num_heads 8 \
    --mlp_ratio 2.0 \
    --drop 0.1 \
    --attn_drop 0.1 \
    --lr 5e-4 \
    --weight_decay 0.01 \
    --dataset both \
    > log/log_v2_42_5e4.log \
    2> log/err_v2_42_5e4.log
