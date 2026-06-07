#!/bin/bash

conda activate team15

mkdir -p log

declare -l CHECK_REPRODUCIBILITY

CHECK_REPRODUCIBILITY=$2

if [ "$CHECK_REPRODUCIBILITY" -eq "true" ]; then
    LOOP_START=1
    LOOP_END=3
else
    LOOP_START=0
    LOOP_END=0
fi

SEED=44

for TIME in {${LOOP_START}..${LOOP_END}}; do
    python3 main.py \
        --seed ${SEED} \
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
        --time_stamp ${TIME} \
        > log/log_v3_42_both.log \
        2> log/err_v3_42_both.log
done

if [ ${LOOP_START} -ne 0 ]; then
    for TIME in {${LOOP_START}..${LOOP_END}}; do
        python3 image.py \
            --seed ${SEED} \
            --ckpt_path cache/best_both_c68_${TIME}.pt \
            --dataset both \
            --num_images 3 \
            --output output/both_${SEED}_${TIME}.png
    done
else
    python3 image.py \
            --seed ${SEED} \
            --ckpt_path cache/best_both_c68.pt \
            --dataset both \
            --num_images 3 \
            --output output/both_${SEED}_${TIME}.png
fi