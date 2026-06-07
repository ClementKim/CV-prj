#!/bin/bash

conda activate team15

CUDA_VISIBLE_DEVICES=1

SEED=44
TIME=2

python3 main.py \
    --seed ${SEED} \
    --batch 32 \
    --epochs 1 \
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
    > log/log_v3_${SEED}_both_${TIME}.log \
    2> log/err_v3_${SEED}_both_${TIME}.log

python3 image.py \
    --seed ${SEED} \
    --ckpt_path cache/best_both_c68_${TIME}.pt \
    --dataset both \
    --num_images 3 \
    --output output/both_${SEED}_${TIME}.png