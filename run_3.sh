#!/bin/bash
# Improved model: ImageNet-pretrained dual encoder (timm ViT-S/16 + torchvision ResNet-34),
# train-only augmentation, AMP, NaN-skip + grad clip, differential LR, best-val checkpoint.
# Requires `pip install timm` and internet on the first run (downloads pretrained weights).
# If the GPU OOMs at batch 16 (ViT-S @512 is the heavy part), drop to --batch 8.
conda activate CV

for SEED in 42 43 44;
do
    # for TIME in {1..3};
    for TIME in {1..1};
    do
        python3 main.py \
            --seed ${SEED} \
            --batch 32 \
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
            > log/log_v3_${SEED}_both_${TIME}.log \
            2> log/err_v3_${SEED}_both_${TIME}.log
    done
done

for MODEL in unet unet++ fpn pspnet deeplabv3 deeplabv3+ linknet manet pan upernet pan segformer dpt;
do 
    python3 baseline.py \
        --model ${MODEL} \
        --seed 42 \
        --batch 16 \
        --epochs 30 \
        --warmup_epochs 2 \
        --lr 3e-4 \
        --weight_decay 0.05 \
        --weight_pow 0.5 \
        --max_class_weight 10.0 \
        --workers 8 \
        --dataset both \
        > log/log_${MODEL}_42_both.log \
        2> log/err_${MODEL}_42_both.log
done

git add log

git commit -m "add log for run_3.sh"

git push