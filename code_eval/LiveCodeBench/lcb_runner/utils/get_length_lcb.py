import json
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm
import os


def count_tokens_in_outputs(file_path, model_name):
    """
    Read a JSON file with specified structure and count tokens in output_list[0]
    
    Args:
        file_path (str): Path to the JSON file
        model_name (str): HuggingFace model name to use for tokenization
    
    Returns:
        dict: Statistics about token counts
    """
    # Load the tokenizer
    print(f"Loading tokenizer for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load the JSON file
    print(f"Loading data from: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("Expected JSON data to be a list")
    
    # Count tokens for each entry
    token_counts = []
    print("Counting tokens...")
    
    for entry in tqdm(data):
        if 'output_list' in entry and len(entry['output_list']) > 0:
            output_text = entry['output_list'][0]
            tokens = tokenizer.encode(output_text)
            token_count = len(tokens)
            
            # Store the count along with the question ID for reference
            token_counts.append({
                'question_id': entry.get('question_id', 'unknown'),
                'token_count': token_count
            })
    
    # Calculate statistics
    if token_counts:
        counts = [item['token_count'] for item in token_counts]
        total_tokens = sum(counts)
        avg_tokens = total_tokens / len(counts)
        max_tokens = max(counts)
        min_tokens = min(counts)
        
        stats = {
            'total_entries': len(token_counts),
            'total_tokens': total_tokens,
            'average_tokens': avg_tokens,
            'max_tokens': max_tokens,
            'min_tokens': min_tokens,
            'token_counts': token_counts
        }
    else:
        stats = {
            'total_entries': 0,
            'total_tokens': 0,
            'average_tokens': 0,
            'max_tokens': 0,
            'min_tokens': 0,
            'token_counts': []
        }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Count tokens in output_list[0] of JSON entries')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the JSON file')
    parser.add_argument('--model_name', type=str, required=True, help='HuggingFace model name for tokenization')
    parser.add_argument('--output_file', type=str, default=None, help='Path to save the token statistics (JSON)')
    
    args = parser.parse_args()
    
    stats = count_tokens_in_outputs(args.file_path, args.model_name)
    
    # Print summary statistics
    print("\nToken Count Statistics:")
    print(f"Total entries analyzed: {stats['total_entries']}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Average tokens per entry: {stats['average_tokens']:.2f}")
    print(f"Maximum tokens: {stats['max_tokens']}")
    print(f"Minimum tokens: {stats['min_tokens']}")
    
    # Save statistics to file if requested
    if args.output_file:
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=4)
        print(f"\nStatistics saved to: {args.output_file}")

if __name__ == "__main__":
    main()
    


