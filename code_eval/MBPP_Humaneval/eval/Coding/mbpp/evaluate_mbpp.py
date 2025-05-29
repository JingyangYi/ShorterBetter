import json
from datasets import Dataset
import pandas as pd
import torch
from tqdm import tqdm
import os
import torch
import argparse
from vllm import LLM, SamplingParams
import time
import re
import builtins

os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import sys
sys.path.append("./scripts/eval")

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="./eurus-7b-kto-hf")
parser.add_argument("--input_data", type=str, default="./new_mbpp.json")
parser.add_argument("--save_dir", type=str, default="./")
parser.add_argument("--model_type", type=str, default='mistral')

args = parser.parse_args()

os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# STOP_WORDS =["\nassert", "assert", "\ndef "]


def generate_sample_batch(question_list):
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
    )
    sampling_params = SamplingParams(max_tokens=32768,
                                    temperature=0.9,
                                    n=1,
                                    )

    outputs = llm.generate(question_list, sampling_params, use_tqdm=True)
    completions = []
    raw_outputs = []
    for output in outputs:
        raw_text = output.outputs[0].text
        raw_outputs.append(raw_text)
        # Extract code block if present
        code_match = re.search(r"```python\n(.*?)\n```", raw_text, re.DOTALL)
        if code_match:
            completion = code_match.group(1).strip()
        else:
            completion = raw_text.split('```')[0].strip()
        completions.append(completion)
    return completions, raw_outputs

def make_signature(code):
    signature = [line for line in code.split("\n") if line.strip().startswith("def ")][0]
    signature = signature.lstrip("def ").replace(" ", "").rstrip(":").strip().replace(",", ", ")
    assert ":" not in signature
    return signature

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)
def make_conv(signature, description, test_list):
    description = description.split(" https://www.")[0]
    #testcase = "\n>>> ".join(test_list)
    testcase = test_list[0]
    prompt = (
                f"Write Python code to solve the task.\n"
                f"Write a Python function `{signature}` to solve the following problem:"
                f"{description}\n"
                f">>> {testcase}\n"
                f"Present your code in ```python\nYOUR CODE HERE```"
            )
    msg =  [{"role": "user", "content": prompt}]

    out = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    # out = out.lstrip(tokenizer.bos_token).strip()
    # out = out.rstrip(tokenizer.eos_token).strip()
    return out
import contextlib
import signal
class TimeoutException(Exception):
    pass
@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

def exec_helper(code):
    with time_limit(3):
        exec(compile(code, filename="mbpp", mode='exec'), globals())

def evaluate(dataset):
    correct = 0
    format_error = 0
    exec_error = 0

    for example in dataset.to_dict(orient="records"):
        completion = example["completion"]
        # Ensure the completion starts with 'def'
        if not completion.strip().startswith("def"):
            completion = "def " + completion.strip()

        function = completion
        test_cases = "\n".join(example["test_list"]).replace("\/", "/")
        test_run = "\n".join([
            function,
            test_cases,
        ])

        try:
            exec_helper(function)
        except Exception as e:
            format_error += 1
            continue

        try:
            # run test case
            exec_helper(test_cases)
            exec_helper(test_run)
        except:
            exec_error += 1
            continue
        else:
            correct += 1
    return 100 * correct / len(dataset), 100 * exec_error / len(dataset), 100 * format_error / len(dataset)


def calculate_token_stats(raw_outputs):
    """Calculate token statistics for the raw outputs"""
    token_counts = []
    
    for output in raw_outputs:
        tokens = tokenizer.encode(output)
        token_counts.append(len(tokens))
    
    # Use a different variable name than 'sum' to avoid shadowing the built-in function
    import builtins
    total_tokens = builtins.sum(token_counts) if token_counts else 0
    avg_tokens = total_tokens / len(token_counts) if token_counts else 0
    
    return {
        "token_counts": token_counts,
        "average_tokens": avg_tokens,
        "total_samples": len(token_counts),
        "min_tokens": min(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0
    }

if __name__ == "__main__":

    dataset = pd.read_json(args.input_data, lines=False)
    dataset["signature"] = dataset.apply(lambda row: make_signature(row["code"]), axis=1)
    # for signature in dataset["signature"]:
    #     STOP_WORDS.append("\n\nprint(" + signature.split("(")[0].strip())
    dataset["prompt"] = dataset.apply(lambda row: make_conv(row["signature"], row["prompt"], row["test_list"]), axis=1)
    completions, raw_outputs = generate_sample_batch(dataset["prompt"].tolist())
    dataset["completion"] = completions
    dataset["raw_output"] = raw_outputs
    del dataset["source_file"]
    dataset.to_json(os.path.join(args.save_dir, "mbpp_completion.json"))
    
    results = []
    for example in dataset.to_dict(orient="records"):
        completion = example["completion"]
        # Ensure the completion starts with 'def'
        if not completion.strip().startswith("def"):
            completion = "def " + completion.strip()

        function = completion
        test_cases = "\n".join(example["test_list"]).replace("\/", "/")
        test_run = "\n".join([
            function,
            test_cases,
        ])

        # Make sure result is created as a mutable dictionary
        temp_result = {"passed": False, "error_type": None}
        
        try:
            exec_helper(function)
        except Exception:
            temp_result["error_type"] = "format_error"
        else:
            try:
                # run test case
                exec_helper(test_cases)
                exec_helper(test_run)
            except Exception:
                temp_result["error_type"] = "execution_error"
            else:
                temp_result["passed"] = True
                
        # Append the result dictionary to results
        results.append(temp_result)

    # Store all prompt-response pairs with test results in a separate JSON file
    prompt_response_pairs = []
    for i, result in enumerate(results):
        row_data = dataset.iloc[i]  # Get the row data directly by index
        prompt_response_pairs.append({
            "task_id": i,
            "prompt": row_data["prompt"],
            "raw_response": row_data["raw_output"],
            "passed": result["passed"],
            "error_type": result["error_type"]
        })
    
    with open(os.path.join(args.save_dir, "mbpp_prompt_responses.json"), "w") as f:
        json.dump(prompt_response_pairs, f, indent=4)

    # Calculate token statistics and save to a separate JSON file
    token_stats = calculate_token_stats(raw_outputs)
    
    # Add token count for each sample
    output_with_tokens = []
    for i, row in dataset.iterrows():
        output_with_tokens.append({
            "task_id": i,
            "raw_output": row["raw_output"],
            "token_count": token_stats["token_counts"][i]
        })
    
    # Save the token statistics
    token_stats_data = {
        "output_token_counts": output_with_tokens,
        "summary": {
            "average_tokens": token_stats["average_tokens"],
            "total_samples": token_stats["total_samples"],
            "min_tokens": token_stats["min_tokens"],
            "max_tokens": token_stats["max_tokens"]
        }
    }
    
    with open(os.path.join(args.save_dir, "mbpp_token_stats.json"), "w") as f:
        json.dump(token_stats_data, f, indent=2)
    
    # Now calculate the overall statistics
    correct = builtins.sum(1 for r in results if r["passed"])
    format_error = builtins.sum(1 for r in results if r["error_type"] == "format_error")
    exec_error = builtins.sum(1 for r in results if r["error_type"] == "execution_error")
    total = len(results)
    
    accuracy = 100 * correct / total
    format_error_pct = 100 * format_error / total
    exec_error_pct = 100 * exec_error / total

    with open(os.path.join(args.save_dir, "result.txt"), "w") as f:
        result = {
            "accuracy": accuracy, 
            "exec_error": exec_error_pct, 
            "format_error": format_error_pct,
            "average_tokens": token_stats["average_tokens"]
        }
        print(result)
        print(result, file=f)
        result = {
            "accuracy": accuracy, 
            "exec_error": exec_error, 
            "format_error": format_error,
            "average_tokens": token_stats["average_tokens"]
        }
        print(result)
        print(result, file=f)