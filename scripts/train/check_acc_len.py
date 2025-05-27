import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple, Dict, Optional
import os
from scipy.stats import linregress

def analyze_output_log(log_file_path: str, prompts_per_step: int = 64, responses_per_prompt: int = 8, max_steps: Optional[int] = None) -> None:
    """
    Analyze RL training output log to calculate average accuracy and output lengths per step.
    
    Args:
        log_file_path: Path to the output.log file
        prompts_per_step: Number of prompts per step (default: 64)
        responses_per_prompt: Number of responses per prompt (default: 8)
        max_steps: Maximum number of steps to plot (default: None, plots all steps)
    
    Returns:
        None (displays plots)
    """
    # Storage for parsed data
    steps_data = []
    current_step_data = []
    total_responses_per_step = prompts_per_step * responses_per_prompt
    
    # Pattern to extract length and correct information
    pattern = r"Lengths=\[(.*?)\], Correct=\[(.*?)\]"
    
    # Read the log file
    with open(log_file_path, 'r') as file:
        for line_num, line in enumerate(file):
            # Skip non-data lines
            if not "Lengths=" in line:
                continue
                
            # Extract lengths and correct items
            match = re.search(pattern, line)
            if match:
                lengths_str = match.group(1)
                correct_str = match.group(2)
                
                # Parse tensor values for lengths
                lengths = []
                for part in lengths_str.split(', '):
                    if 'tensor' in part:
                        # Extract the number from tensor(X)
                        tensor_match = re.search(r'tensor\((\d+)\)', part)
                        if tensor_match:
                            lengths.append(int(tensor_match.group(1)))
                
                # Parse tensor values for correct items (these are indices, not lengths)
                correct = []
                if correct_str:
                    for part in correct_str.split(', '):
                        if 'tensor' in part:
                            tensor_match = re.search(r'tensor\((\d+)\)', part)
                            if tensor_match:
                                correct.append(int(tensor_match.group(1)))
                
                # Add to current step data
                current_step_data.append({
                    'lengths': lengths,
                    'correct_count': len(correct)
                })
                
                # If we've collected data for a full step, process it
                if len(current_step_data) == prompts_per_step:
                    # Calculate total correct responses for this step
                    total_correct = sum(item['correct_count'] for item in current_step_data)
                    
                    # Calculate accuracy
                    accuracy = total_correct / total_responses_per_step
                    
                    # Calculate length statistics
                    all_lengths = [length for item in current_step_data for length in item['lengths']]
                    avg_length = np.mean(all_lengths) if all_lengths else 0
                    median_length = np.median(all_lengths) if all_lengths else 0
                    
                    # Store step data
                    steps_data.append({
                        'step': len(steps_data) + 1,
                        'accuracy': accuracy,
                        'avg_length': avg_length,
                        'median_length': median_length,
                        'total_correct': total_correct,
                        'possible_correct': total_responses_per_step
                    })
                    
                    # Check if we've reached max_steps
                    if max_steps and len(steps_data) >= max_steps:
                        break
                    
                    # Reset for next step
                    current_step_data = []
    
    # Handle any remaining incomplete step (if exists)
    if current_step_data and (not max_steps or len(steps_data) < max_steps):
        total_correct = sum(item['correct_count'] for item in current_step_data)
        possible_correct = len(current_step_data) * responses_per_prompt
        accuracy = total_correct / possible_correct if possible_correct > 0 else 0
        
        all_lengths = [length for item in current_step_data for length in item['lengths']]
        avg_length = np.mean(all_lengths) if all_lengths else 0
        median_length = np.median(all_lengths) if all_lengths else 0
        
        steps_data.append({
            'step': len(steps_data) + 1,
            'accuracy': accuracy,
            'avg_length': avg_length,
            'median_length': median_length,
            'total_correct': total_correct,
            'possible_correct': possible_correct
        })
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(steps_data)
    
    if df.empty:
        print("No data was successfully parsed from the log file.")
        return
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot accuracy
    ax1.plot(df['step'], df['accuracy'], 'skyblue', marker='o', label='Accuracy')
    
    # Add a fitted line to the accuracy plot
    z = np.polyfit(df['step'], df['accuracy'], 1)  # Linear fit
    p = np.poly1d(z)
    ax1.plot(df['step'], p(df['step']), "r--", label='Fitted Line')
    
    
    ax1.set_ylabel('Accuracy (correct/total responses)')
    ax1.set_title(f'GRPO Training Performance Metrics')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot lengths
    # ax2.plot(df['step'], df['avg_length'], 'r-', marker='o', label='Avg Length')
    ax2.plot(df['step'], df['median_length'], 'r-', marker='o', label='Median Length')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Output Length')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_name = f"training_metrics_plot_{len(df)}steps.png"
    plot_path = os.path.join(os.path.dirname(log_file_path), plot_name)
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    
    # Display the plot
    plt.show()
    
    # Save the processed data
    csv_name = f"training_metrics_{len(df)}steps.csv"
    csv_path = os.path.join(os.path.dirname(log_file_path), csv_name)
    df.to_csv(csv_path, index=False)
    print(f"Metrics data saved to {csv_path}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total Steps Analyzed: {len(df)}")
    print(f"Average Accuracy: {df['accuracy'].mean():.4f}")
    print(f"Average Output Length: {df['avg_length'].mean():.2f}")
    print(f"Final Accuracy: {df['accuracy'].iloc[-1]:.4f}")
    print(f"Final Output Length: {df['avg_length'].iloc[-1]:.2f}")

# Example usage
if __name__ == "__main__":
    log_file_path = "/path/to/output.log"
    analyze_output_log(log_file_path, prompts_per_step=128, responses_per_prompt=8, max_steps=500)
    # prompt_per_step = Your batch size, response_per_prompt = number of rollouts