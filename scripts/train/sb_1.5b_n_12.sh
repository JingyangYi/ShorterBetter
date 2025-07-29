#!/bin/bash
#SBATCH --job-name=sb_1.5b_n_12
#SBATCH --partition=general          
#SBATCH --nodes=1                   
#SBATCH --ntasks=1                   
#SBATCH --gres=gpu:a100:8              
#SBATCH --cpus-per-task=16         
#SBATCH --mem=512G                
#SBATCH --time=12:00:00            
#SBATCH --output=/net/scratch/jiazhengw/ShorterBetter/logs/slurm_logs/verl_grpo/%x_%j.out      
#SBATCH --error=/net/scratch/jiazhengw/ShorterBetter/logs/slurm_logs/verl_grpo/%x_%j.err  
set -x

# Configuration through environment variables
export PROJECT_HOME="/net/scratch/jiazhengw/ShorterBetter"
export LOG_DIR="/net/scratch/jiazhengw/ShorterBetter/logs"
export WANDB_API_KEY="906a7d5d10486eab174334f7df2f209605dc1260"
export DATASET_DIR="/net/scratch/jiazhengw/ShorterBetter/deepscaler/data"

# ----------------------------------------
# This script uses the original naive reward function
# Reward function: naive (original implementation)
# ----------------------------------------

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

MODEL_PATH=${MODEL_PATH:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${DATASET_DIR}/train_filtered.parquet \
    data.val_files=${DATASET_DIR}/train_filtered.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    data.max_prompt_length=1800 \
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
    actor_rollout_ref.rollout.n=12 \
    +actor_rollout_ref.rollout.n_val=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='ShorterBetter_1.5b' \
    trainer.experiment_name='sb_1.5b_n_12' \
    +trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    ++trainer.test_freq=-1 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=1 \
    reward_model.reward_manager=naive "${@:1}" 