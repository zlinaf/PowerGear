#!/bin/bash
DATASET_NAME="atax"
CUDA_NUM=0
##SET PARSER##
SEED_NUM=1
K=5
VAL_INDEX=4
###If you want to set personal parameters, you can change the default value in main.py###
cd ..
CUDA_VISIBLE_DEVICES=$CUDA_NUM python main.py --test_dataset $DATASET_NAME --k $K --fold_index $VAL_INDEX

