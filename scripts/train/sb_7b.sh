#!/bin/bash
#SBATCH --job-name=sb_train_job
#SBATCH --partition=general          
#SBATCH --nodes=1                   
#SBATCH --ntasks=1                   
#SBATCH --gres=gpu:a100:8              
#SBATCH --cpus-per-task=16         
#SBATCH --mem=256G                
#SBATCH --time=12:00:00            
#SBATCH --output=${LOG_DIR}/slurm_logs/verl_grpo/%x_%j.out      
#SBATCH --error=${LOG_DIR}/slurm_logs/verl_grpo/%x_%j.err  
set -x

# Configuration through environment variables
# Set these variables before running:
export PROJECT_HOME="/path/to/project"
export LOG_DIR="/path/to/logs"
export WANDB_API_KEY="your_wandb_api_key"
export DATASET_DIR="${PROJECT_HOME}/deepscaler/data"

# ----------------------------------------
# To change the reward function hyperparameters, please change the alpha and beta in the following:
# /ShorterBetter/verl/verl/workers/reward_manager/naive.py line 244
# By default, alpha=2.0, beta=0.001
# ----------------------------------------

# ----------------------------------------
# The training process will print out the output lengths and correct counts for each batch.
# You can use check_acc_len.py to plot the accuracy and output length trends.
# ----------------------------------------


# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
export PYTHONPATH="${PROJECT_HOME}:$PYTHONPATH"
export VLLM_ATTENTION_BACKEND=XFORMERS

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Default model path if not specified
MODEL_PATH=${MODEL_PATH:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"}

# Train over a single node, 8 A100-80GB GPUs.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${DATASET_DIR}/train_filtered.parquet \
    data.val_files=${DATASET_DIR}/aime.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    data.max_prompt_length=200 \
    data.max_response_length=5000 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    +actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.9 \
    +actor_rollout_ref.rollout.val_temperature=0.9 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=4 \
    +actor_rollout_ref.rollout.n_val=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='ShorterBetter' \
    trainer.experiment_name='sb_7b' \
    +trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    ++trainer.test_freq=-1 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=1 "${@:1}"\

# Notes on the training cofig:

# 1. data.train_batch_size=128; can be change to 64

# 2. data.train_files=${DATASET_DIR}/train_filtered.parquet; Here _filtered means the datapoints are filtered by the data.max_prompt_length=1500

# 3. data.val_files=${DATASET_DIR}/aime.parquet; But we don't evaluate during training, this is just a placeholder.

# 4. actor_rollout_ref.rollout.n=8; can be changed to 4 but the length reduction process will be slower.

# 5. actor_rollout_ref.actor.optim.lr=1e-6 is recommended.

# 6. during 7B training, we sometimes encounter OOM error when trying to save the model checkpoint. 
# If this happens, you can try to set the total steps it trains to avoid saving checkpoints by limiting the total datapoints in the training set.