#!/bin/bash
# Usage: bash math_eval.sh <model_path> <model_name>

MODEL_PATH=$1
MODEL_NAME=$2

export HF_HOME=/net/scratch/jiazhengw/huggingface

python /net/scratch/jiazhengw/ShorterBetter/scripts/eval/math/math_eval.py \
    --model_path "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --batch_size 16 \
    --tasks all