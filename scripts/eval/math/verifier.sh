#!/bin/bash
#SBATCH --job-name=llm_verify
#SBATCH --partition=general          
#SBATCH --nodes=1                   
#SBATCH --ntasks=1                   
#SBATCH --gres=gpu:1               
#SBATCH --cpus-per-task=16          
#SBATCH --mem=128G                  
#SBATCH --time=12:00:00            
#SBATCH --output=/path/to/logs/%x_%j.out      
#SBATCH --error=/path/to/logs/%x_%j.err  

export HF_HOME=/path/to/.cache/huggingface
data_dir="/ShorterBetter/eval_data/outputs/math/DeepSeek-R1-Distill-Qwen-1.5B"


# By default, the output will be save in OUTPUT_DIR = "/ShorterBetter/eval_data/outputs/math"
python verifier.py \
    --dataset_dir $data_dir \
    --output_dir $data_dir/verified \
    --batch_size 16
