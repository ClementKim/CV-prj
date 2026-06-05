#!/bin/bash

conda activate CV

# for SEED in 42 43 44;
for SEED in 44;
do
    for MODEL in unet unet++ fpn pspnet deeplabv3 deeplabv3+ linknet manet pan upernet segformer dpt;
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
    done
done