#!/bin/bash
# Usage: bash verifier.sh <data_dir>

DATA_DIR=$1

export HF_HOME=/net/scratch/jiazhengw/huggingface

python /net/scratch/jiazhengw/ShorterBetter/scripts/eval/math/verifier.py \
    --dataset_dir "$DATA_DIR" \
    --output_dir "$DATA_DIR/verified" \
    --batch_size 16