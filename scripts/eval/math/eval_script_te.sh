#!/bin/bash
#SBATCH --job-name=eval_math
#SBATCH --partition=general          
#SBATCH --nodes=1                   
#SBATCH --ntasks=1                   
#SBATCH --gres=gpu:1               
#SBATCH --cpus-per-task=16         
#SBATCH --mem=256G                  
#SBATCH --time=12:00:00            
#SBATCH --output=/net/scratch/jiazhengw/ShorterBetter/logs/eval_logs/%x_%j.out      
#SBATCH --error=/net/scratch/jiazhengw/ShorterBetter/logs/eval_logs/%x_%j.err 

# Usage: bash eval_script.sh
# This script will loop through a list of checkpoint names.

set -e

CHECKPOINT_LIST=("te_1.5b_opt")
BASE_DIR="/net/scratch/jiazhengw/ShorterBetter"

for CHECKPOINT_NAME in "${CHECKPOINT_LIST[@]}"; do
    echo "Processing checkpoint: $CHECKPOINT_NAME"
    CHECKPOINT_PATH="$BASE_DIR/checkpoints/training_efficient/${CHECKPOINT_NAME}/global_step_200/actor/huggingface"
    DATA_DIR="$BASE_DIR/eval_data/outputs/math/${CHECKPOINT_NAME}"

    # 1. Merge models
    echo "Merging models for $CHECKPOINT_NAME..."
    python $BASE_DIR/verl/scripts/model_merger.py \
        --local_dir $BASE_DIR/checkpoints/training_efficient/${CHECKPOINT_NAME}/global_step_200/actor

    # 2. Evaluate models
    echo "Evaluating models for $CHECKPOINT_NAME..."
    bash $BASE_DIR/scripts/eval/math/math_eval.sh "$CHECKPOINT_PATH" "$CHECKPOINT_NAME"

    # 3. Verify results
    echo "Verifying results for $CHECKPOINT_NAME..."
    bash $BASE_DIR/scripts/eval/math/verifier.sh "$DATA_DIR"

    echo "Workflow complete for $CHECKPOINT_NAME!"
    echo "----------------------------------------"
done

echo "All workflows complete!"