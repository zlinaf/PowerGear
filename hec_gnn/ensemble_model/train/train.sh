#!/bin/bash
DATASET_NAME_SET=("atax" "bicg" "gemm" "gesummv" "k2mm" "k3mm" "mvt" "syr2k" "syrk")
CUDA_NUM=0
##SET PARSER##
SEED_NUM=(1 2 3)
K=10
VAL_INDEX=(0 1 2 3 4 5 6 7 8 9)
###If you want to set personal parameters, you can change the default value in main.py###
for dataset in $DATASET_NAME_SET;do
    echo $dataset
    for seed in $SEED_NUM;do
        echo $seed
        for fold_index in $VAL_INDEX;do
            echo $fold_index
            CUDA_VISIBLE_DEVICES=$CUDA_NUM python main.py --test_dataset $dataset --k $K --fold_index $fold_index --seed $seed
        done;
    done;
done;
