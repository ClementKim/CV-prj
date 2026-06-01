#!/bin/bash

conda activate CV

python3 main.py \
    --seed 42 \
    --batch 4 \
    --epochs 10 \
    --patch_size 16 \
    --embed_dim 256 \
    --num_heads 8 \
    --mlp_ratio 2.0 \
    --drop 0.1 \
    --attn_drop 0.1 \
    --lr 5e-5 \
    --weight_decay 0.01 \
    > log/log_42_256 \
    2> log/err_42_256

python3 main.py \
    --seed 42 \
    --batch 4 \
    --epochs 10 \
    --patch_size 16 \
    --embed_dim 512 \
    --num_heads 8 \
    --mlp_ratio 2.0 \
    --drop 0.1 \
    --attn_drop 0.1 \
    --lr 5e-5 \
    --weight_decay 0.01 \
    > log/log_42_512 \
    2> log/err_42_512

python3 main.py \
    --seed 43 \
    --batch 4 \
    --epochs 10 \
    --patch_size 16 \
    --embed_dim 256 \
    --num_heads 8 \
    --mlp_ratio 2.0 \
    --drop 0.1 \
    --attn_drop 0.1 \
    --lr 5e-5 \
    --weight_decay 0.01 \
    > log/log_43_256 \
    2> log/err_43_256

python3 main.py \
    --seed 43 \
    --batch 4 \
    --epochs 10 \
    --patch_size 16 \
    --embed_dim 512 \
    --num_heads 8 \
    --mlp_ratio 2.0 \
    --drop 0.1 \
    --attn_drop 0.1 \
    --lr 5e-5 \
    --weight_decay 0.01 \
    > log/log_43_512 \
    2> log/err_43_512

python3 main.py \
    --seed 44 \
    --batch 4 \
    --epochs 10 \
    --patch_size 16 \
    --embed_dim 256 \
    --num_heads 8 \
    --mlp_ratio 2.0 \
    --drop 0.1 \
    --attn_drop 0.1 \
    --lr 5e-5 \
    --weight_decay 0.01 \
    > log/log_44_256 \
    2> log/err_44_256

python3 main.py \
    --seed 44 \
    --batch 4 \
    --epochs 10 \
    --patch_size 16 \
    --embed_dim 512 \
    --num_heads 8 \
    --mlp_ratio 2.0 \
    --drop 0.1 \
    --attn_drop 0.1 \
    --lr 5e-5 \
    --weight_decay 0.01 \
    > log/log_44_512 \
    2> log/err_44_512