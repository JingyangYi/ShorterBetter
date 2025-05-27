"""Script to prepare DeepScaler training and test datasets.

This script processes math problem datasets into a standardized format for training
and testing DeepScaler models. It loads problems from specified datasets, adds
instruction prompts, and saves the processed data as parquet files.
"""

import argparse
import os
from typing import Dict, List, Optional, Any

import pandas as pd
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed
import sys
sys.path.append('../..')
from deepscaler.data.utils import load_dataset
from deepscaler.data.dataset_types import TrainDataset, TestDataset

import random
import copy



def extract_solution(solution_str: str) -> str:
    """Extract the final boxed solution from a solution string.

    Args:
        solution_str: Raw solution string that may contain multiple boxed answers

    Returns:
        The final boxed answer with box notation removed
    """
    return remove_boxed(last_boxed_only_string(solution_str))


def make_map_fn(split: str):
    """Create a mapping function to process dataset examples.

    Args:
        split: Dataset split name ('train' or 'test')

    Returns:
        Function that processes individual dataset examples
    """
    
    def process_fn(example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        question = example.pop('problem')
        instruction = "\n\nGive the final answer within \\boxed{}."
        
        question = f"{question} {instruction}"
        answer = example.pop('answer')

        data = {
            "data_source": "",
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer,
            },
            "extra_info": {
                'split': split,
                'index': idx
            }
        }
        return data
    return process_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process datasets for DeepScaler training')
    parser.add_argument('--local_dir', default='/net/scratch/jingyang/SB2/verl/deepscaler/data',
                       help='Local directory to save processed datasets')
    parser.add_argument('--hdfs_dir', default=None,
                       help='Optional HDFS directory to copy datasets to')
    parser.add_argument('--num_tokens', default=-1, type=int,
                       help='Number of tokens to think for')
    parser.add_argument('--use_log', default=False, action='store_true',
                       help='Use log scale for number of tokens')
    parser.add_argument('--use_both', default=False, action='store_true',
                       help='Use both normal and token constraints')
    parser.add_argument('--use_both_both', default=False, action='store_true',
                       help='Use both max budget and token constraints')
    parser.add_argument('--do_normal', default=False, action='store_true',
                       help='Do normal prompt, without any thinking limit')
    parser.add_argument('--sample_multiple_for_train', default=1, type=int,
                       help='Sample multiple examples for train')
                                              
    args = parser.parse_args()

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    NUM_TOKENS = args.num_tokens
    USE_LOG = args.use_log
    USE_BOTH = args.use_both
    USE_BOTH_BOTH = args.use_both_both
    USE_NORMAL = args.do_normal
    if USE_NORMAL:
        assert not USE_LOG and not USE_BOTH and not USE_BOTH_BOTH and NUM_TOKENS == -1
    SAMPLE_MULTIPLE_FOR_TRAIN = args.sample_multiple_for_train
    # if NUM_TOKENS != -1:
    #     local_dir = local_dir+'_'+str(NUM_TOKENS)
    # Make local directory if it doesn't exist
    makedirs(local_dir, exist_ok=True)

    # Initialize datasets
    train_datasets = [TrainDataset.DEEPSCALER]
    train_dataset = load_dataset(train_datasets[0])
    test_datasets = [TestDataset.AIME, TestDataset.AMC, TestDataset.MATH, TestDataset.MINERVA, TestDataset.OLYMPIAD_BENCH]
    
    test_datasets_data = [load_dataset(d) for d in test_datasets]

    # Process training data
    train_data: List[Dict[str, Any]] = []
    process_fn = make_map_fn('train')
    train_dataset_original = copy.deepcopy(train_dataset)
    for i in range(SAMPLE_MULTIPLE_FOR_TRAIN):
        train_dataset = copy.deepcopy(train_dataset_original)
        for idx, example in enumerate(train_dataset):
            processed_example = process_fn(example, idx)
            if processed_example is not None:
                train_data.append(processed_example)

    # Process and save each test dataset separately
    for test_dataset, test_data_list in zip(test_datasets, test_datasets_data):
        test_data: List[Dict[str, Any]] = []
        process_fn = make_map_fn('test')
        for idx, example in enumerate(test_data_list):
            processed_example = process_fn(example, idx)
            if processed_example is not None:
                test_data.append(processed_example)

        dataset_name = test_dataset.value.lower()
        test_df = pd.DataFrame(test_data)
        test_df.to_parquet(os.path.join(local_dir, f'{dataset_name}.parquet'))
        print(f"{dataset_name} test data size:", len(test_data))

    # Save training dataset
    print("train data size:", len(train_data))
    train_df = pd.DataFrame(train_data)
    train_df.to_parquet(os.path.join(local_dir, 'train.parquet'))

    # Optionally copy to HDFS
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)