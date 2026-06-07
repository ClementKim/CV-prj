#!/bin/bash

conda activate CV

SEED=44
python3 main.py \
    --seed ${SEED} \
    --batch 16 \
    --epochs 20 \
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
    > log/log_v3_${SEED}_both.log \
    2> log/err_v3_${SEED}_both.log

python3 image.py \
    --seed ${SEED} \
    --ckpt_path cache/best_both_c68.pt \
    --dataset both \
    --num_images 3 \
    --output output/both_${SEED}.png

for MODEL in unet unet++ fpn deeplabv3 deeplabv3+ linknet manet pan segformer;
do 
    python3 baseline.py \
        --model ${MODEL} \
        --seed ${SEED} \
        --batch 16 \
        --epochs 20 \
        --warmup_epochs 2 \
        --lr 3e-4 \
        --weight_decay 0.05 \
        --weight_pow 0.5 \
        --max_class_weight 10.0 \
        --workers 8 \
        --dataset both \
        > log/log_${MODEL}_${SEED}_both.log \
        2> log/err_${MODEL}_${SEED}_both.log

    python3 image.py \
        --seed ${SEED} \
        --ckpt_path cache/best_both_c68_${MODEL}.pt \
        --dataset both \
        --num_images 3 \
        --output output/both_${MODEL}_${SEED}.png
done