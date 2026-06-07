#!/bin/bash

conda activate team15

CUDA_VISIBLE_DEVICES=3

SEED=44
TIME=1

for MODEL in unet unet++;
do 
    python3 baseline.py \
        --model ${MODEL} \
        --seed ${SEED} \
        --batch 32 \
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
        --ckpt_path cache/best_both_c68_${TIME}.pt \
        --dataset both \
        --num_images 3 \
        --output output/both_${SEED}_${TIME}.png
done