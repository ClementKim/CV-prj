#!/bin/bash

conda env list | awk '{print $1}' | grep -qx team15 && ENV_EXISTS=0 || ENV_EXISTS=1

if [ $ENV_EXISTS -ne 0 ]; then
    echo "Starting environment setup"

    conda create -n team15 python=3.10.14

    conda activate team15
    
    pip install "torch==2.10.0" "torchvision==0.25.0" timm numpy pillow matplotlib tqdm segmentation_models_pytorch

    echo "Environment setup complete"
fi

# Checking the existance of tar files
ls | grep -qx MINC.tar && DATASET_EXISTS1=0 || DATASET_EXISTS1=1
ls | grep -qx OPENSURFACE.tar && DATASET_EXISTS2=0 || DATASET_EXISTS2=1
ls | grep -qx minc && DATASET_EXISTS3=0 || DATASET_EXISTS3=1
ls | grep -qx photo_orig && DATASET_EXISTS4=0 || DATASET_EXISTS4=1
ls | grep -qx opensurfaces-data && DATASET_EXISTS5=0 || DATASET_EXISTS5=1

if [ $DATASET_EXISTS1 -ne 0 ] && [ $DATASET_EXISTS2 -ne 0 ] && [ $DATASET_EXISTS3 -ne 0 ] && [ $DATASET_EXISTS4 -ne 0 ] && [ $DATASET_EXISTS5 -ne 0 ]; then
    echo "Download tar files from https://drive.google.com/drive/folders/1BJFKH0cRnT5ZZMy1MzJZQfKx1G0y6-C0?usp=sharing"

else
    if [ $DATASET_EXISTS1 -eq 0 ]; then
        if [ $DATASET_EXISTS3 -ne 0 ] && [ $DATASET_EXISTS4 -ne 0 ]; then
            echo "Extracting MINC dataset"

            tar -xvf MINC.tar

            if [ $? -ne 0 ]; then
                echo "Failed to extract MINC dataset. Please check the tar file and try again."
                exit 1
            
            else
                echo "MINC dataset extracted successfully"
                rm -rf MINC.tar
            fi
        fi
    fi

    if [ $DATASET_EXISTS2 -eq 0 ]; then
        if [ $DATASET_EXISTS5 -ne 0 ]; then
            echo "Extracting opensurface dataset"

            mkdir -p opensurfaces-data

            mv OPENSURFACE.tar opensurfaces-data/

            cd opensurfaces-data

            tar -xvf OPENSURFACE.tar

            if [ $? -ne 0 ]; then
                echo "Failed to extract opensurface dataset. Please check the tar file and try again."
                exit 1
            
            else
                echo "opensurface dataset extracted successfully"
                rm -rf OPENSURFACE.tar
            fi
        fi
    fi
fi