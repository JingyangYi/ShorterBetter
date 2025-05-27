#!/bin/bash
#SBATCH --job-name=eval_ood
#SBATCH --partition=general          
#SBATCH --nodes=1                   
#SBATCH --ntasks=1                   
#SBATCH --gres=gpu:1               
#SBATCH --cpus-per-task=16          
#SBATCH --mem=256G                  
#SBATCH --time=12:00:00            
#SBATCH --output=/path/to/logs/%x_%j.out      
#SBATCH --error=/path/to/logs/%x_%j.err  

export HF_HOME=/path/to/.cache/huggingface


# ----------------------------------------
# After evaluation, please use bbh_eval.py and ood_mul_choice_verifier.py to verify the results.
# ----------------------------------------

python ood_eval.py \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --model_name DeepSeek-R1-Distill-Qwen-1.5B \
    --batch_size 16
