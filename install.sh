#!/bin/bash

conda env list | awk '{print $1}' | grep -qx team15 && ENV_EXISTS=0 || ENV_EXISTS=1

if [ $? -ne 0 ]; then
    ls -al | awk '{print $9}' | grep -qx team15 && ENV_EXISTS=0 || ENV_EXISTS=1
fi

if [ $ENV_EXISTS -ne 0 ]; then
    echo "Starting environment setup"

    conda create -n team15 python=3.10.14

    if [ $? -ne 0 ]; then
        python -m venv team15
        source team15/bin/activate
        pip install --upgrade pip

    else
        conda activate team15

    fi

    pip install torch torchvision timm numpy pillow

    echo "Environment setup complete"
fi

ls | grep -qx opensurfaces-data && DATASET_EXISTS1=0 || DATASET_EXISTS1=1
ls | grep -qx MINC && DATASET_EXISTS2=0 || DATASET_EXISTS2=1

if [ $DATASET_EXISTS1 -ne 0  && $DATASET_EXISTS2 -ne 0]; then
    BOTH=0
else
    BOTH=1
fi

if [ $BOTH -eq 0 ]; then
    if [ $DATASET_EXISTS1 -ne 0 ]; then
        echo "Download Opensurface dataset"

        mkdir -p opensurfaces-data
        cd opensurfaces-data

        wget http://labelmaterial.s3.amazonaws.com/release/opensurfaces-release-0.zip
        wget http://labelmaterial.s3.amazonaws.com/release/process_opensurfaces_release_0.py

        unzip opensurfaces-release-0.zip

        python3 process_opensurfaces_release_0.py

        cd ..
    fi

    if [ $DATASET_EXISTS2 -ne 0 ]; then
        echo "Download MINC dataset"

        tar -xvf MINC.tar

        if [ $? -ne 0 ]; then
            echo "Failed to extract MINC dataset. Please check the tar file and try again."
            exit 1
        fi 
    fi