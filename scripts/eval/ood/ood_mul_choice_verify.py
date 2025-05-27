import os
import json
import re
from transformers import AutoTokenizer
from tqdm import tqdm

# Load the tokenizer once
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

def extract_answer(response):
    # First check for boxed expressions (keeping existing logic)
    # First try simple boxed expressions without nesting
    simple_matches = re.findall(r"\\boxed\{([^{}]+)\}", response)
    if simple_matches:
        return simple_matches[-1]
    
    # If no simple matches, try the more complex pattern with limited nesting
    nested_matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)+)\}", response)
    if nested_matches:
        return nested_matches[-1]
    
    # Manual approach for boxed expressions as a last resort
    boxed_start = response.find("\\boxed{")
    if boxed_start >= 0:
        start = boxed_start + 7  # Skip past "\boxed{"
        brace_count = 1
        end = start
        
        while end < len(response) and brace_count > 0:
            if response[end] == '{':
                brace_count += 1
            elif response[end] == '}':
                brace_count -= 1
            end += 1
            
        if brace_count == 0:  # Found matching brace
            return response[start:end-1]
    
    # New patterns for various answer formats
    
    # Pattern 1: "Final Answer: X" or "Answer: X" or "ANSWER: X"
    answer_patterns = [
        r"(?:final\s+answer|answer|ANSWER)\s*:\s*([a-eA-E](?:\s*\)|\.)?|(?:\s*\)?\s*)?([0-9]+|[^,\n\.]*))",
        r"(?:final\s+answer|answer|ANSWER)\s*:\s*([a-eA-E])\s*\)?\s*(?:\\?\(?\s*\\?)?([^,\n\.]*)",
        r"(?:the\s+(?:best|correct)\s+answer\s+is)\s*(?:\*\*)?\s*([a-eA-E])(?:\.\s*|\s+)([^,\n\.]*)"
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            # Take the last match (most likely the final answer)
            match = matches[-1]
            if isinstance(match, tuple):
                # Return the letter choice if present, otherwise the answer text
                return match[0].strip() if match[0].strip() else match[1].strip()
            else:
                return match.strip()
    
    # Look for standalone letter choices with formatting
    letter_choice = re.search(r"\*\*\s*([a-eA-E])\s*(?:\.|:|\))\s*\*\*", response)
    if letter_choice:
        return letter_choice.group(1)
    
    # Look for "The answer is X" pattern
    answer_is = re.search(r"the\s+answer\s+is\s+(?:\*\*)?\s*([a-eA-E]|[0-9]+)(?:\*\*)?", response, re.IGNORECASE)
    if answer_is:
        return answer_is.group(1)
    
    return None

def process_json_files(base_dir="ood", task_result=""):
    # List all directories in the base directory
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for dir_name in dirs:
        dir_path = os.path.join(base_dir, dir_name)
        
        # Specify the JSON file to process
        json_file = f"{task_result}_results.json"
        file_path = os.path.join(dir_path, json_file)
        
        # Check if the specified JSON file exists
        if not os.path.exists(file_path):
            print(f"No {json_file} found in {dir_path}, skipping")
            continue
        
        # Read the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Process each entry
        for entry in data:
            if 'completion' in entry and 'target_answer' in entry:
                # Extract answer from completion
                choices = entry['problem'].split('\nHere are the choices:\n')[1].lower().split('\n')
                
                if len(choices) == 1:
                    choices = choices[0].split(', ')
                
                extracted_answer = str(extract_answer(entry['completion'])).lower() # abcde or string
                target_answer = entry['target_answer'].lower() # abcde
                target_answer_index = 0 if target_answer == 'a' else 1 if target_answer == 'b' else 2 if target_answer == 'c' else 3 if target_answer == 'd' else 4
                target_answer_string = choices[target_answer_index] # string

                # Compare with target_answer and update pass_or_fail
                if (extracted_answer == target_answer or 
                    extracted_answer == target_answer_string or
                    re.search(r'\b' + re.escape(extracted_answer) + r'\b', target_answer_string)):
                    entry['pass_or_fail'] = True
                else:
                    entry['pass_or_fail'] = False
        
        # Save the updated JSON file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"Processed {file_path}")

def update_summary_files(base_dir="ood", task_result=""):
    # List all directories in the base directory
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for dir_name in tqdm(dirs, desc="Processing directories"):
        dir_path = os.path.join(base_dir, dir_name)
        summary_path = os.path.join(dir_path, 'summary.json')
        
        # Initialize or load summary dictionary
        if os.path.exists(summary_path):
            print(f"Loading existing summary.json in {dir_path}")
            # Read the existing summary.json
            with open(summary_path, 'r') as f:
                summary = json.load(f)
        else:
            print(f"No summary.json found in {dir_path}, creating new one")
            summary = {}
        
        # Specify the JSON file to process
        json_file = f"{task_result}_results.json"
        file_path = os.path.join(dir_path, json_file)
        
        # Check if the specified JSON file exists
        if not os.path.exists(file_path):
            print(f"No {json_file} found in {dir_path}, skipping")
            continue
        
        # Read the JSON file
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                print(f"Warning: {file_path} does not contain a list, skipping")
                continue
            
            # Calculate metrics
            total_problems = len(data)
            correct_count = sum(1 for entry in data if entry.get('pass_or_fail', False))
            accuracy = correct_count / total_problems if total_problems > 0 else 0
            
            # Calculate average output length using tokenizer
            output_lengths = []
            for entry in data:
                if 'completion' in entry:
                    completion = entry.get('completion', '')
                    # Use tokenizer to measure token count
                    token_count = tokenizer(completion, return_tensors="pt").input_ids.shape[1]
                    output_lengths.append(token_count)
            
            avg_output_length = sum(output_lengths) / len(output_lengths) if output_lengths else 0
            
            # Add or update the task in summary
            if task_result not in summary:
                # Create new entry for task
                summary[task_result] = {
                    "accuracy": accuracy,
                    "avg_output_length": avg_output_length,
                    "correct_count": correct_count,
                    "total_problems": total_problems
                }
            else:
                # Update existing entry
                summary[task_result]['correct_count'] = correct_count
                summary[task_result]['accuracy'] = accuracy
                summary[task_result]['avg_output_length'] = avg_output_length
                summary[task_result]['total_problems'] = total_problems
            
            print(f"Processed {task_result} for {dir_name}: accuracy={accuracy:.4f}, avg_length={avg_output_length:.2f} tokens")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        
        # Save the updated or newly created summary.json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"{'Updated' if os.path.exists(summary_path) else 'Created'} summary.json for {dir_name}")

if __name__ == "__main__":
    dir_path = "/ShorterBetter/eval_data/outputs/ood"
    for task_result in ["mmlu", "mathqa"]:
        process_json_files(dir_path, task_result)
        update_summary_files(dir_path, task_result)
