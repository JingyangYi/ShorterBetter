import os
import json
import argparse
from vllm import LLM, SamplingParams
from tqdm import tqdm
import torch

# Verification prompt template from sample_generation.py
verification_prompt_template = """You are a mathematical answer validation system. Your task is to determine if two mathematical expressions are equivalent.

Are these answers mathematically equivalent? Consider these factors:
1. Different forms of the same number (fractions, decimals, scientific notation)
2. Algebraic equivalence (e.g., polynomial factorization and simplification)
3. Trigonometric equivalence
4. Logical equivalence for boolean expressions
5. Equivalence of sets, vectors, or matrices
6. Tolerance for small rounding errors in numerical answers

**Task:**  
Answer with only `"True"` if the expressions are equivalent or `"False"` if they are not.

**Example**
Target Answer: 2.0
Extracted Answer: 2
Your response: True

Target Answer: 0.5
Extracted Answer: 1/2
Your response: True

Target Answer: k = 2
Extracted Answer: 2
Your response: True

Target Answer: 3.14159
Extracted Answer: π
Your response: True

Target Answer: x^2 - 4
Extracted Answer: (x-2)(x+2)
Your response: True

Target Answer: 2y + 2z
Extracted Answer: 2(y+z)
Your response: True

Target Answer: sin^2(θ) + cos^2(θ)
Extracted Answer: 1
Your response: True

Target Answer: tan(x)
Extracted Answer: sin(x)/cos(x)
Your response: True

Target Answer: √(x²)
Extracted Answer: |x|
Your response: True

Target Answer: 1 + 1/2 + 1/4 + 1/8 + ...
Extracted Answer: 2
Your response: True

Target Answer: lim(x→0) sin(x)/x
Extracted Answer: 1
Your response: True

Target Answer: x² + 2x + 1
Extracted Answer: (x+1)²
Your response: True

**Input:**  
Target Answer: {target}  
Extracted Answer: {extracted}  
Your response:
"""

gpqa_verification_prompt = """
You are a mathematical and scientific answer validation system. Your task is to determine if two expressions are equivalent.

Are these answers semantically equivalent? Consider these factors:
1. Different forms of the same number (fractions, decimals, scientific notation)
2. Algebraic equivalence (e.g., polynomial factorization and simplification)
3. Trigonometric equivalence
4. Logical equivalence for boolean expressions
5. Equivalence of sets, vectors, or matrices
6. Tolerance for small rounding errors in numerical answers
7. Scientific answers that convey the same meaning even with different wording
8. Partial answers that correctly identify the main entity or concept asked

**Task:**  
Answer with only `"True"` if the expressions are equivalent or `"False"` if they are not.

**Example**
Target Answer: 2.0
Extracted Answer: 2
Your response: True

Target Answer: 0.5
Extracted Answer: 1/2
Your response: True

Target Answer: k = 2
Extracted Answer: 2
Your response: True

Target Answer: 3.14159
Extracted Answer: π
Your response: True

Target Answer: x^2 - 4
Extracted Answer: (x-2)(x+2)
Your response: True

Target Answer: 2y + 2z
Extracted Answer: 2(y+z)
Your response: True

Target Answer: sin^2(θ) + cos^2(θ)
Extracted Answer: 1
Your response: True

Target Answer: tan(x)
Extracted Answer: sin(x)/cos(x)
Your response: True

Target Answer: √(x²)
Extracted Answer: |x|
Your response: True

Target Answer: 1 + 1/2 + 1/4 + 1/8 + ...
Extracted Answer: 2
Your response: True

Target Answer: lim(x→0) sin(x)/x
Extracted Answer: 1
Your response: True

Target Answer: x² + 2x + 1
Extracted Answer: (x+1)²
Your response: True

Target Answer: Planet_1 is preferred due to its ~1.65 times higher probability to transit.
Extracted Answer: Planet_1
Your response: True

Target Answer: weaker - slower
Extracted Answer: \\text{Weaker oxidant, stronger oxidant}
Your response: True

Target Answer: The electric field decreases as 1/r² from a point charge.
Extracted Answer: E ∝ 1/r²
Your response: True

Target Answer: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune
Extracted Answer: Mercury through Neptune in order of distance from the Sun
Your response: True

Target Answer: Sodium chloride dissolves in water because the polar water molecules separate the Na+ and Cl- ions.
Extracted Answer: NaCl is soluble due to ion-dipole interactions with water
Your response: True

Target Answer: Hybridization of carbon in methane (CH₄) is sp³
Extracted Answer: sp³
Your response: True

**Input:**  
Target Answer: {target}  
Extracted Answer: {extracted}  
Your response:

"""

def extract_answer(response):
    """Extract the answer from the boxed expression in the completion"""
    import re
    
    # First try simple boxed expressions without nesting
    simple_matches = re.findall(r"\\boxed\{([^{}]+)\}", response)
    if simple_matches:
        return simple_matches[-1]
    
    # If no simple matches, try the more complex pattern with limited nesting
    nested_matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)+)\}", response)
    if nested_matches:
        return nested_matches[-1]
    
    # Manual approach as a last resort
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
    
    return None

def verify_responses(model_name, dataset_dir, output_dir=None, batch_size=16, gpqa=False):
    """Verify the correctness of math responses using LLM verification with batched inference"""
    # Load LLM model for verification
    print(f"Loading verification model: {model_name}")
    llm = LLM(model=model_name, dtype="bfloat16", max_model_len=8000,
              tensor_parallel_size=torch.cuda.device_count())  # Adjust tensor_parallel_size as needed
    
    sampling_params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=128)
    
    # Find all JSON files in the dataset directory
    if not gpqa:
        json_files = [f for f in os.listdir(dataset_dir) if f.endswith('.json') and not f.startswith('summary')]
        prompt_template = verification_prompt_template
    else:
        json_files = ["gpqa_results.json"]
        prompt_template = gpqa_verification_prompt
    
    # Initialize or load summary dictionary
    summary_path = os.path.join(output_dir or dataset_dir, "summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as f:
                summary = json.load(f)
        except Exception as e:
            print(f"Error loading existing summary, creating new one: {e}")
            summary = {}
    else:
        summary = {}
    
    # Process each JSON file
    for file_name in json_files:
        file_path = os.path.join(dataset_dir, file_name)
        output_path = os.path.join(output_dir or dataset_dir, f"verified_{file_name}")
        
        print(f"Processing {file_path}")
        
        # Load the file
        with open(file_path, 'r') as f:
            problems = json.load(f)
        
        if not isinstance(problems, list):
            print(f"Warning: {file_path} is not a list, skipping")
            continue

        # Track changes for verification
        total_problems = len(problems)
        verified_count = 0
        correct_count = 0
        
        # Process problems in batches
        for i in tqdm(range(0, len(problems), batch_size)):
            batch = problems[i:i+batch_size]
            
            # Prepare batch prompts and track indices of valid problems
            batch_prompts = []
            valid_indices = []
            
            for j, problem in enumerate(batch):
                # Skip if no target_answer is available
                if 'target_answer' not in problem:
                    continue
                    
                target_answer = str(problem.get('target_answer', '')).strip()
                completion = problem.get('completion', '')
                
                # Extract the answer from the completion
                extracted_answer = extract_answer(completion)
                
                if extracted_answer is None:
                    # If no answer was extracted, mark as incorrect
                    problem['pass_or_fail'] = False
                    continue
                
                # Create verification prompt
                prompt = prompt_template.replace("{target}", target_answer).replace("{extracted}", extracted_answer)
                batch_prompts.append(prompt)
                valid_indices.append(i + j)  # Store the global index
            
            if not batch_prompts:
                continue  # Skip if no valid prompts in this batch
            
            # Run batch inference
            outputs = llm.generate(batch_prompts, sampling_params=sampling_params)
            
            # Process batch results
            for k, output in enumerate(outputs):
                problem_idx = valid_indices[k]
                verification_result = output.outputs[0].text.strip().lower() if output.outputs else "false"
                
                # Update problem with verification result
                is_correct = "true" in verification_result
                problems[problem_idx]['pass_or_fail'] = is_correct
                
                verified_count += 1
                if is_correct:
                    correct_count += 1
            
            # Save progress periodically
            if (i + batch_size) % 100 == 0 or (i + batch_size) >= len(problems):
                print(f"Progress: {verified_count}/{total_problems} problems verified")
                with open(output_path, 'w') as f:
                    json.dump(problems, f, indent=2)
        
        # Save final results
        with open(output_path, 'w') as f:
            json.dump(problems, f, indent=2)
        
        print(f"Verification completed for {file_name}")
        print(f"Total problems: {total_problems}")
        print(f"Verified problems: {verified_count}")
        print(f"Correct problems: {correct_count}")
        accuracy = correct_count / total_problems if total_problems > 0 else 0
        print(f"Accuracy: {accuracy:.2f}")
        
        # Always update or create summary entry
        # Extract dataset name from file_name (removing _results.json part)
        dataset_name = file_name.replace("_results.json", "")
        
        # Create or update the summary entry
        if dataset_name not in summary:
            summary[dataset_name] = {}
            
        summary[dataset_name]["accuracy"] = accuracy
        summary[dataset_name]["correct_count"] = correct_count
        summary[dataset_name]["total_count"] = total_problems
        summary[dataset_name]["verified_count"] = verified_count
        
        # Save the updated summary
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Updated summary for {dataset_name}")
        
    return summary

def main():
    parser = argparse.ArgumentParser(description="Verify math responses using LLM verification")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B", help="Verification model name")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing JSON files to verify")
    parser.add_argument("--output_dir", type=str, help="Directory to save verified results (defaults to dataset_dir)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--gpqa", type=bool, default=False, help="verify gpqa dataset?")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    verify_responses(args.model, args.dataset_dir, args.output_dir, args.batch_size, args.gpqa)

if __name__ == "__main__":
    main()