import os
import re
import argparse
import pandas as pd
from datasets import Dataset
import json # Added import for json operations
import torch # Added import (likely needed by vllm/torch.cuda)

# Ensure correct relative path if running from a different directory
import sys
# Assuming the script is run from the same directory it resides in,
# and 'utils' is two levels up. Adjust if needed.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# It's safer to import after adjusting sys.path
try:
    from utils.data import write_jsonl, read_problems, HUMAN_EVAL
    from utils.evaluation import evaluate_functional_correctness
except ImportError as e:
    print(f"Error importing utils: {e}")
    print("Please ensure the script is placed correctly relative to the 'utils' directory"
          " or adjust the sys.path.append line.")
    sys.exit(1)


from vllm import LLM, SamplingParams
from transformers import AutoTokenizer # Moved import to top


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="Path to the model directory or name.")
parser.add_argument("--save_dir", help="Directory to save samples and results.")
parser.add_argument("--num-samples-per-task", type=int, default=1, help="Number of samples to generate per task (used by sampling_params n).") # Note: n=1 is hardcoded below currently
# for pass@1
# https://github.com/bigcode-project/bigcode-evaluation-harness/blob/c326b51eef25f96ca9b8d22300612b64f3253992/docs/README.md?plain=1#L44
parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature.")
# Add argument for problem file path if needed, defaulting to HUMAN_EVAL
parser.add_argument("--problem_file", type=str, default=HUMAN_EVAL, help="Path to the HumanEval problem file.")
# Add arguments for evaluation if they differ from defaults
parser.add_argument("--k", type=str, default="1", help="Comma-separated list of k values for pass@k calculation.") # Defaulting to pass@1 based on temp=0.2
parser.add_argument("--n_workers", type=int, default=4, help="Number of workers for evaluation.")
parser.add_argument("--timeout", type=float, default=10.0, help="Timeout per problem for evaluation.") # Increased default timeout slightly


args = parser.parse_args()

# --- Ensure save directory exists ---
os.makedirs(args.save_dir, exist_ok=True)

# --- Load Tokenizer ---
# Load the tokenizer globally once
try:
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True) # Added trust_remote_code
    print(f"Tokenizer loaded successfully from {args.model}")
    # Set padding token if not set, common practice for generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer pad_token set to eos_token.")
except Exception as e:
    print(f"Error loading tokenizer from {args.model}: {e}")
    sys.exit(1)


problems = read_problems(args.problem_file)
# https://github.com/bigcode-project/bigcode-evaluation-harness/blob/c326b51eef25f96ca9b8d22300612b64f3253992/bigcode_eval/tasks/humaneval.py#L54C13-L54C87


def generate_sample_batch(question_list):
    """Generates completions and returns both raw outputs and extracted code."""
    try:
        llm = LLM(
            model=args.model,
            trust_remote_code=True,
            tensor_parallel_size=torch.cuda.device_count()
        )
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        sys.exit(1)

    # Adjust sampling_params n based on args if needed, although pass@1 usually means n=1
    sampling_params = SamplingParams(
        max_tokens=32768,
        temperature=args.temperature,
        n=1,
        stop=["<|eot_id|>"] if tokenizer.eos_token == "<|eot_id|>" else [tokenizer.eos_token] # Use tokenizer's eos
    )
    print(f"Using stop token(s): {sampling_params.stop}")
    
    print(f"Generating {len(question_list)} samples...")
    outputs = llm.generate(question_list, sampling_params, use_tqdm=True) # Changed use_tqdm to True

    raw_outputs_text = [output.outputs[0].text for output in outputs]
    
    completions = []
    for text in raw_outputs_text:
        if '```' in text:
            parts = re.split(r'```python|```', text)
            if len(parts) > 2:  # Ensure there is content between ```python and ```
                completion_code = parts[1].strip()  # Extract the part between ```python and ```
            else:
                completion_code = text.strip()  # Fallback to the entire text if no match
            completions.append(completion_code)
        else:
            completions.append(text.strip())


    # --- Calculate Token Lengths ---
    token_lengths = [len(tokenizer.encode(raw_text)) for raw_text in raw_outputs_text]

    return completions, raw_outputs_text, token_lengths # Return raw text and lengths too

def make_signature(example):
    """Extracts function signature."""
    match = re.search(
            rf"def\s+({example['entry_point']}.*?):\s*\n", example["prompt"]
        )
    if match:
        return match.group(1)
    else:
        # Fallback or error handling if signature not found
        print(f"Warning: Could not find signature for entry point {example['entry_point']} in prompt.")
        return example['entry_point'] # Return entry point as a basic fallback

def make_conv(example):
    """Formats the prompt for the chat model."""
    signature = make_signature(example) # Reuse signature extraction
    docstring_match = re.search(r"(?:\"\"\"|''')(.*?)(?:\"\"\"|''')", example["prompt"], re.DOTALL)

    # Revised prompt structure based on original logic
    prompt = (
        f"Write Python code to solve the task.\n"
        f"Write a Python function `{signature}` to solve the following problem: Present code in ```python```\n"
        f"```python\n"
        f"{example['prompt']}"
        f"```\n"
    )

    # Apply chat template
    # Ensure tokenizer is available in this scope (it is, as it's loaded globally)
    msg = [{"role": "user", "content": prompt}]
    try:
        # Use the globally loaded tokenizer
        formatted_prompt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        print(f"Error applying chat template: {e}")
        # Fallback if template application fails (might happen with unexpected models)
        formatted_prompt = prompt # Use the raw formatted prompt as fallback


    return formatted_prompt


def entry_point(
    sample_file: str,
    k_str: str = "1,10,100", # Renamed variable to avoid conflict with loop var
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k_list = list(map(int, k_str.split(","))) # Use the renamed variable
    print(f"Evaluating for pass@{k_list}...")
    results = evaluate_functional_correctness(sample_file, k=k_list, n_workers=n_workers, timeout=timeout, problem_file=problem_file)
    results = {f"pass@{k_val}":v*100 for k_val,v in results.items()} # Use k_val for clarity
    print("Evaluation Results:")
    print(results)
    # Optionally save results to a file
    results_filepath = sample_file.replace(".jsonl", "_results.json")
    try:
        with open(results_filepath, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Evaluation results saved to: {results_filepath}")
    except Exception as e:
        print(f"Error saving evaluation results: {e}")


# --- Main Execution ---

# Load problems into a dataset
problems_df = pd.DataFrame(problems).T
problems_dataset = Dataset.from_pandas(problems_df)

# --- Prepare prompts ---
print("Preparing prompts...")
# Define cache file paths relative to save_dir
sig_cache_path = os.path.join(args.save_dir, "cache-human_eval-sig.arrow")
conv_cache_path = os.path.join(args.save_dir, "cache-human_eval-conv.arrow")

# Use cache files for faster re-runs if prompts don't change
problems_dataset = problems_dataset.map(
    lambda x: {"signature": make_signature(x)},
    cache_file_name=sig_cache_path,
    load_from_cache_file=os.path.exists(sig_cache_path) # Load if exists
)
problems_dataset = problems_dataset.map(
    lambda x: {"instruction": make_conv(x)},
    cache_file_name=conv_cache_path,
    load_from_cache_file=os.path.exists(conv_cache_path) # Load if exists
)
print("Prompts prepared.")

# --- Generate Samples ---
completions, raw_outputs, token_lengths = generate_sample_batch(problems_dataset["instruction"])

# --- Process and Save Token Length Information ---
if token_lengths:
    average_length = sum(token_lengths) / len(token_lengths)
    print(f"\nAverage output token length (raw model output): {average_length:.2f}")
else:
    average_length = 0
    print("\nNo token lengths generated.")

# Prepare data for JSON
length_data = {
    "individual_lengths": token_lengths,
    "average_length": average_length,
    "model": args.model, # Add model info for context
    "temperature": args.temperature, # Add temp info
}

# Define JSON file path for lengths
length_filepath = os.path.join(args.save_dir, "output_token_lengths.json")

# Save to JSON
try:
    with open(length_filepath, 'w') as f:
        json.dump(length_data, f, indent=4)
    print(f"Token length information saved to: {length_filepath}")
except Exception as e:
    print(f"Error saving token length information: {e}")

# --- Prepare Samples for Evaluation ---
# Add the *extracted* completions to the dataset
problems_dataset = problems_dataset.add_column("completion", completions)

problems_dataset = problems_dataset.map(lambda x: {"completion": x["completion"]})

# Convert to list of dictionaries for saving
samples_list = []
# Need to reconstruct the original 'problems' dictionary structure with task_id
task_ids = list(problems.keys())
for i, record in enumerate(problems_dataset):
     task_id = task_ids[i] # Get the original task_id
     sample = {
         "task_id": task_id,
         "prompt": problems[task_id]["prompt"], # Use original prompt
         "entry_point": problems[task_id]["entry_point"],
         "completion": record["completion"], # Use processed completion
     }
     samples_list.append(sample)


# Save samples
output_filepath = os.path.join(args.save_dir, "samples.jsonl")
write_jsonl(output_filepath, samples_list)
print(f"Generated samples saved to: {output_filepath}")

# --- Run Evaluation ---
# Call evaluation function using args for parameters
entry_point(
    sample_file=output_filepath,
    k_str=args.k,
    n_workers=args.n_workers,
    timeout=args.timeout,
    problem_file=args.problem_file
)
