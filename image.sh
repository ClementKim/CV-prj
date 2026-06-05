#!/bin/bash

deactivate

conda activate CV

pip install matplotlib

python3 image.py \
    --ckpt_path \
    --dataset both \
    --num_images 6