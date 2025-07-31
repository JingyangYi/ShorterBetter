#!/usr/bin/env python3
"""
Script to calculate the average output length from JSON files in the evaluation data directory.
"""

import json
import os
import argparse
from typing import List, Dict, Any
import numpy as np

def calculate_average_output_length(data_dir: str) -> None:
    """
    Calculate the average output length from all JSON files in the specified directory.
    
    Args:
        data_dir: Path to the directory containing JSON files
    """
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} does not exist.")
        return
    
    json_files = []
    all_output_lengths = []
    
    # Find all JSON files in the directory
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            json_files.append(os.path.join(data_dir, filename))
    
    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files in {data_dir}")
    
    # Process each JSON file
    file_stats = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract output lengths from the JSON data
            # The structure might vary, so we'll try different common patterns
            output_lengths = []
            
            if isinstance(data, list):
                # If data is a list of responses
                for item in data:
                    if isinstance(item, dict):
                        # Try different possible field names for output length
                        length_found = False
                        for field in ['output_length', 'length', 'response_length', 'output_len']:
                            if field in item:
                                output_lengths.append(item[field])
                                length_found = True
                                break
                        # If no specific length field, try to calculate from response text
                        if not length_found and ('response' in item or 'output' in item or 'answer' in item):
                            text = item.get('response', item.get('output', item.get('answer', '')))
                            if isinstance(text, str):
                                output_lengths.append(len(text.split()))
            elif isinstance(data, dict):
                # If data is a single response or has a specific structure
                for field in ['output_length', 'length', 'response_length', 'output_len']:
                    if field in data:
                        output_lengths.append(data[field])
                        break
                # Check if there's a responses field
                if 'responses' in data and isinstance(data['responses'], list):
                    for response in data['responses']:
                        if isinstance(response, dict):
                            for field in ['output_length', 'length', 'response_length', 'output_len']:
                                if field in response:
                                    output_lengths.append(response[field])
                                    break
            
            all_output_lengths.extend(output_lengths)
            
            # Calculate per-file statistics
            if output_lengths:
                file_avg = np.mean(output_lengths)
                file_median = np.median(output_lengths)
                file_min = np.min(output_lengths)
                file_max = np.max(output_lengths)
                file_std = np.std(output_lengths)
                
                file_stats.append({
                    'filename': os.path.basename(json_file),
                    'count': len(output_lengths),
                    'average': float(file_avg),
                    'median': float(file_median),
                    'min': int(file_min),
                    'max': int(file_max),
                    'std': float(file_std)
                })
                
                print(f"Processed {os.path.basename(json_file)}: {len(output_lengths)} samples, avg: {file_avg:.2f}, median: {file_median:.2f}, min: {file_min}, max: {file_max}")
            else:
                print(f"Processed {os.path.basename(json_file)}: found 0 output lengths")
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    if not all_output_lengths:
        print("No output lengths found in any JSON files.")
        return
    
    # Calculate statistics
    avg_length = np.mean(all_output_lengths)
    median_length = np.median(all_output_lengths)
    min_length = np.min(all_output_lengths)
    max_length = np.max(all_output_lengths)
    std_length = np.std(all_output_lengths)
    
    print(f"\nOutput Length Statistics:")
    print(f"Total samples: {len(all_output_lengths)}")
    print(f"Average length: {avg_length:.2f}")
    print(f"Median length: {median_length:.2f}")
    print(f"Min length: {min_length}")
    print(f"Max length: {max_length}")
    print(f"Standard deviation: {std_length:.2f}")
    
    # Save results to a file
    results = {
        'total_samples': len(all_output_lengths),
        'average_length': float(avg_length),
        'median_length': float(median_length),
        'min_length': int(min_length),
        'max_length': int(max_length),
        'std_length': float(std_length),
        'all_lengths': all_output_lengths,
        'per_file_statistics': file_stats
    }
    
    output_file = os.path.join(data_dir, 'output_length_statistics.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Calculate average output length from JSON files.')
    parser.add_argument('data_dir', type=str, help='Directory containing JSON files')
    
    args = parser.parse_args()
    calculate_average_output_length(args.data_dir)

if __name__ == "__main__":
    main() 