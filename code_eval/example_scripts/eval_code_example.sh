#!/bin/bash

# MBPP_Humaneval

MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
echo "model name: $MODEL"
temperature=0.9

OUTPUT_DIR=./code_eval/MBPP_Humaneval/eval_cache

cd ./code_eval/MBPP_Humaneval/eval/Coding/human_eval
echo "running human_eval evaluation"
mkdir -p $OUTPUT_DIR/humaneval
python ./code_eval/MBPP_Humaneval/eval/Coding/human_eval/evaluate_human_eval.py \
  --model $MODEL \
  --save_dir $OUTPUT_DIR/humaneval_instruct/ \
  --num-samples-per-task 1 \
  --temperature $temperature
  
cd ./code_eval/MBPP_Humaneval/eval/Coding/mbpp
echo "running mbpp evaluation"
mkdir -p $OUTPUT_DIR/mbpp
python evaluate_mbpp.py \
  --model $MODEL \
  --input_data 	new_mbpp.json \
  --save_dir $OUTPUT_DIR/mbpp_instruct_7B \

# LiveCodeBench

MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MODEL_NAME=$(basename $MODEL) 
SCENARIO=codegeneration
MAX_TOKENS=16289
RELEASE_VERSION=release_v5
codegen_n=1
temperature=0.9
n=1
STOP_WORDS="None"

cd ./code_eval/LiveCodeBench
source .venv/bin/activate

python -m lcb_runner.runner.main --model $MODEL --scenario $SCENARIO --max_tokens $MAX_TOKENS --release_version $RELEASE_VERSION --evaluate --codegen_n $codegen_n --n $n --temperature $temperature --stop $STOP_WORDS
python -m lcb_runner.utils.get_length_lcb --model_name $MODEL --file_path ./output/$MODEL_NAME/Scenario.${SCENARIO}_${codegen_n}_${temperature}.json
