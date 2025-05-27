#!/usr/bin/env python3
import os
import json
import argparse
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
import glob
import time
from transformers import AutoTokenizer
from pathlib import Path

# Paths
DATA_DIR = "/ShorterBetter/eval_data/ood"
OUTPUT_DIR = "ShorterBetter/eval_data/outputs/ood"

# Load model and tokenizer function following generation.py approach
def get_model(model_path):
    """Load model and tokenizer using vLLM with multi-GPU support
    
    Supports:
    1. Standard HuggingFace models (local or from HF Hub)
    2. Models with merged_model.pt format
    3. Direct HuggingFace Hub models by name
    4. Locally trained models with sharded safetensors
    5. Special handling for Marco-o1 models
    """
    print(f"Loading model from: {model_path}")
    
    # Special handling for Marco-o1 models
    if "Marco-o1" in model_path:
        print(f"Detected Marco-o1 model, using AIDC-AI/Marco-o1 tokenizer")
        try:
            tokenizer = AutoTokenizer.from_pretrained("AIDC-AI/Marco-o1", trust_remote_code=True)
            llm = LLM(
                model=model_path,
                tokenizer="AIDC-AI/Marco-o1",
                dtype="bfloat16",
                max_model_len=16000,
                tensor_parallel_size=torch.cuda.device_count(),
                trust_remote_code=True
            )
            print(f"Successfully loaded Marco-o1 model with AIDC-AI/Marco-o1 tokenizer")
            return llm, tokenizer
        except Exception as e:
            print(f"Error loading Marco-o1 model: {e}")
            raise e
    
    # Check if model_path is a HuggingFace Hub model ID (contains '/' but doesn't start with '/')
    is_hf_hub_model = '/' in model_path and not model_path.startswith('/')
    
    if is_hf_hub_model:
        print(f"Detected HuggingFace Hub model ID: {model_path}")
        try:
            # Load directly from HuggingFace Hub
            llm = LLM(
                model=model_path,
                dtype="bfloat16",
                max_model_len=16000,
                tensor_parallel_size=torch.cuda.device_count(),
                trust_remote_code=True  # Needed for some models
            )
            print(f"Model loaded from HF Hub with tensor parallelism across {torch.cuda.device_count()} GPUs")
            
            # Get tokenizer from HF Hub
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            return llm, tokenizer
        except Exception as e:
            print(f"Error loading model from HF Hub: {e}")
            raise e
    
    # Check for sharded safetensors format
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        print(f"Found sharded safetensors model at: {model_path}")
        
        try:
            # Load model with vLLM
            llm = LLM(
                model=model_path,
                dtype="bfloat16",
                max_model_len=16000,
                tensor_parallel_size=torch.cuda.device_count(),
                trust_remote_code=True
            )
            print(f"Model loaded with tensor parallelism across {torch.cuda.device_count()} GPUs")
            
            # Get tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            return llm, tokenizer
        except Exception as e:
            print(f"Error loading sharded safetensors model: {e}")
            raise e
    
    # For local models, check if we have a merged_model.pt file
    merged_model_path = os.path.join(model_path, "merged_model.pt")
    if os.path.exists(merged_model_path):
        print(f"Found merged_model.pt at: {merged_model_path}")
        
        # Get HuggingFace directory for tokenizer and config
        hf_dir = os.path.join(model_path, "huggingface")
        if not os.path.exists(hf_dir):
            raise ValueError(f"HuggingFace directory not found at {hf_dir}")
        
        print(f"Using tokenizer and config from: {hf_dir}")
        
        # Create a temporary directory to prepare HF-compatible files
        import tempfile
        import shutil
        
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Creating temporary HF directory at: {temp_dir}")
            
            # Copy config and tokenizer files
            for file in os.listdir(hf_dir):
                shutil.copy(os.path.join(hf_dir, file), temp_dir)
            
            # Convert merged_model.pt to pytorch_model.bin
            print("Converting merged_model.pt to HF format...")
            
            try:
                # Try to create a symlink first (faster)
                os.symlink(merged_model_path, os.path.join(temp_dir, "pytorch_model.bin"))
                print("Created symlink to merged_model.pt")
            except (OSError, NotImplementedError) as e:
                # If symlink fails, copy the file (slower but more compatible)
                print(f"Symlink failed ({e}), copying file instead (this may take a while)...")
                shutil.copy(merged_model_path, os.path.join(temp_dir, "pytorch_model.bin"))
            
            # Load with vLLM from the temporary directory
            try:
                llm = LLM(
                    model=temp_dir,
                    dtype="bfloat16",
                    max_model_len=16000,
                    tensor_parallel_size=torch.cuda.device_count(),
                    trust_remote_code=True
                )
                
                # Get tokenizer from the original HF directory
                tokenizer = AutoTokenizer.from_pretrained(hf_dir, trust_remote_code=True)
                
                return llm, tokenizer
            except Exception as e:
                print(f"Error loading from temporary directory: {e}")
                print("Trying alternative loading method...")
                
                # If simple symlink/copy doesn't work, we might need to convert the model format
                # This is a placeholder for your custom conversion logic if needed
                raise e
    
    # If no merged_model.pt, try standard HF loading
    try:
        # Check if model has huggingface directory with model weights
        hf_dir = os.path.join(model_path, "huggingface")
        if os.path.exists(hf_dir) and any(f.endswith(('.bin', '.safetensors')) for f in os.listdir(hf_dir)):
            print(f"Using HuggingFace directory with weights: {hf_dir}")
            model_path = hf_dir
        
        # Load model with vLLM
        llm = LLM(
            model=model_path,
            dtype="bfloat16",
            max_model_len=16000,
            tensor_parallel_size=torch.cuda.device_count(),
            trust_remote_code=True
        )
        print(f"Model loaded with tensor parallelism across {torch.cuda.device_count()} GPUs")
        
        # Get tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        return llm, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        
        # Try one more fallback - maybe it's a HF model with a different structure
        try:
            print("Trying to load as a standard HuggingFace model...")
            
            # Try with different parameters
            llm = LLM(
                model=model_path,
                dtype="bfloat16",
                max_model_len=16000,
                tensor_parallel_size=torch.cuda.device_count(),
                trust_remote_code=True,
                revision="main"  # Try specific revision
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True,
                revision="main"
            )
            
            return llm, tokenizer
        except Exception as fallback_e:
            print(f"All loading methods failed. Final error: {fallback_e}")
        
        raise e

def load_dataset(task_name, max_datapoints=None):
    """Load a dataset from DATA_DIR based on task name"""
    if task_name.endswith('.json'):
        dataset_path = os.path.join(DATA_DIR, task_name)
    else:
        dataset_path = os.path.join(DATA_DIR, f"{task_name}.json")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    if max_datapoints is not None:
        dataset = dataset[:max_datapoints]
    
    print(f"Loaded {len(dataset)} problems from {dataset_path}")
    return dataset, dataset_path

def get_all_datasets():
    """Get all available datasets in DATA_DIR"""
    dataset_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    return [os.path.basename(f) for f in dataset_files]

def evaluate_dataset(llm, dataset, dataset_path, model_name, tokenizer, save_path, batch_size=8, task_name=None):
    """Evaluate a model on a dataset using batched inference and save results"""
    sampling_params = SamplingParams(temperature=0.9, top_p=0.9, max_tokens=16000)
    
    results = []
    correct_count = 0
    total_token_count = 0

    prefix_prompt = "Here is a multiple choice question. Choose the best answer from the given options."
    suffix_prompt_1 = "Give your final choice in \\boxed{}."
    suffix_prompt_2 = "Give your final answer in \\boxed{}."
    # Process in batches for efficient multi-GPU inference
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating batches"):
        batch = dataset[i:i+batch_size]
        
        # Prepare batch of prompts
        prompts = []
        for item in batch:
            problem = item['problem']
            # Add the boxed instruction to the problem
            if task_name == "gpqa" or task_name == "bbh":
                prompt = problem + suffix_prompt_2
            else:
                prompt = prefix_prompt + problem + suffix_prompt_1
            prompts.append(prompt)
        
        # Generate responses with vLLM (handles batch processing automatically)
        outputs = llm.generate(prompts, sampling_params=sampling_params)
        
        # Process results
        for j, item in enumerate(batch):
            problem = item['problem']
            target_answer = item['answer']
            
            # Get completion for this problem
            completion = outputs[j].outputs[0].text
            
            # Get output length in tokens
            input_tokens = tokenizer(prompts[j], return_tensors="pt").input_ids.shape[1]
            output_tokens = tokenizer(completion, return_tensors="pt").input_ids.shape[1]
            
            
            total_token_count += output_tokens
            
            # Record result
            result = {
                "problem": problem,
                "target_answer": target_answer,
                "completion": completion,
                "output_length": output_tokens,
            }
            
            results.append(result)
    
    # Calculate accuracy and average output length
    accuracy = correct_count / len(dataset) if len(dataset) > 0 else 0
    avg_output_length = total_token_count / len(dataset) if len(dataset) > 0 else 0
    
    print(f"Dataset evaluation completed:")
    print(f"Average output length: {avg_output_length:.2f} tokens")
    
    # Save results
    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(save_path, f"{os.path.basename(dataset_path).split('.')[0]}_results.json")
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filename}")
    
    return {
        "avg_output_length": avg_output_length,
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate language model on math tasks")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model to evaluate")
    parser.add_argument("--model_name", type=str, required=True, help="Name to use for saving results")
    parser.add_argument("--tasks", type=str, nargs="+", default=["all"], 
                        help="Names of tasks to evaluate (e.g., 'amc', 'aime') or 'all'")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for inference")
    
    args = parser.parse_args()
    
    # Create output directory
    save_dir = os.path.join(OUTPUT_DIR, args.model_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Get list of tasks to evaluate
    if "all" in args.tasks:
        task_files = get_all_datasets()
    else:
        task_files = [f"{task}.json" if not task.endswith('.json') else task for task in args.tasks]
    
    # Load model and tokenizer (only once)
    llm, tokenizer = get_model(args.model_path)
    
    # Evaluate each task
    for task_file in task_files:
        print(f"\n--- Evaluating {task_file} ---")
        
        # Load dataset
        dataset, dataset_path = load_dataset(task_file, max_datapoints=2)
        task_name = os.path.basename(task_file).split('.')[0]
        
        # Run evaluation with batched inference
        result = evaluate_dataset(
            llm=llm,
            dataset=dataset,
            dataset_path=dataset_path,
            model_name=args.model_name,
            tokenizer=tokenizer,
            save_path=save_dir,
            batch_size=args.batch_size,
            task_name=task_name
        )
        

    
    print("Evaluation complete!")
    print(f"To verifier the final answer, please use bbh_eval.py and ood_mul_choice_verifier.py")

if __name__ == "__main__":
    main()