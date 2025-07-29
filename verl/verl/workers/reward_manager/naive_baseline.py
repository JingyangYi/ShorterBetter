# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl import DataProto
import torch
import numpy as np

# --- Helper functions copied from naive.py ---
def verify_correctness(solution_str, ground_truth) -> float:
    retval = False
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)
            if is_equiv(answer, ground_truth):
                retval = True
    except Exception as e:
        print(e)
    return retval

def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False
    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2

def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]
    left = "\\boxed{"
    assert s[:len(left)] == left
    assert s[-1] == "}"
    return s[len(left):-1]

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    return retval

def strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    string = fix_sqrt(string)
    string = string.replace(" ", "")
    string = fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = fix_a_slash_b(string)
    return string

def remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string

# --- End helpers ---

class SigmoidRewardManager:
    """Reward manager using sigmoid length penalty as described in the referenced method section."""
    def __init__(self, tokenizer, num_examine, train_batch_size, num_generation, alpha=1.0, compute_score=None):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.train_batch_size = train_batch_size
        self.num_generation = num_generation
        self.alpha = alpha
        self.compute_score = compute_score

    def check_correctness_and_length(self, data):
        train_batch_size = self.train_batch_size
        num_generation = self.num_generation
        assert len(data) == train_batch_size * num_generation, f"Input data should have len={train_batch_size * num_generation}, got {len(data)}"
        problem_ids = [item.non_tensor_batch['index'] for item in data]
        unique_problem_ids = []
        for pid in problem_ids:
            if pid not in unique_problem_ids:
                unique_problem_ids.append(pid)
        assert len(unique_problem_ids) == train_batch_size, f"Expected {train_batch_size} unique problems, got {len(unique_problem_ids)}"
        correctness_map = {}
        length_map = {}
        group_lengths = {}
        for problem_id in unique_problem_ids:
            indices = [i for i in range(len(data)) if data[i].non_tensor_batch['index'] == problem_id]
            completions_given_prompt = [data[i] for i in indices]
            assert len(completions_given_prompt) == num_generation
            response_lengths_given_prompt = []
            correctnesses_given_prompt = []
            for idx, completion in zip(indices, completions_given_prompt):
                prompt_ids = completion.batch['prompts']
                prompt_length = prompt_ids.shape[-1]
                valid_prompt_length = completion.batch['attention_mask'][:prompt_length].sum()
                response_ids = completion.batch['responses']
                valid_response_length = completion.batch['attention_mask'][prompt_length:].sum()
                response_str = self.tokenizer.decode(response_ids[:valid_response_length], skip_special_tokens=True)
                ground_truth = completion.non_tensor_batch['reward_model']['ground_truth']
                correct_or_not = verify_correctness(response_str, ground_truth)
                correctnesses_given_prompt.append(correct_or_not)
                response_lengths_given_prompt.append(valid_response_length)
                correctness_map[idx] = correct_or_not
                length_map[idx] = valid_response_length
            # Only use correct responses for normalization
            correct_lengths = [l for c, l in zip(correctnesses_given_prompt, response_lengths_given_prompt) if c]
            
            # Print training dynamics (same as naive reward manager)
            print(f"Lengths={response_lengths_given_prompt}, Correct={correct_lengths}", flush=True)
            
            if correct_lengths:
                group_lengths[problem_id] = np.array(correct_lengths, dtype=np.float32)
            else:
                group_lengths[problem_id] = np.array(response_lengths_given_prompt, dtype=np.float32)  # fallback: all
        correctness_set = [correctness_map[i] for i in range(len(data))]
        length_set = [length_map[i] for i in range(len(data))]
        group_lengths_list = [group_lengths[data[i].non_tensor_batch['index']] for i in range(len(data))]
        return correctness_set, length_set, group_lengths_list

    def __call__(self, data: DataProto):
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}
        correctness_set, length_set, group_lengths_list = self.check_correctness_and_length(data)
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        for i in range(len(data)):
            data_item = data[i]
            correct_or_not = correctness_set[i]
            completion_length = length_set[i]
            group_lengths = group_lengths_list[i]
            if correct_or_not:
                mean = np.mean(group_lengths)
                std = np.std(group_lengths) + 1e-7
                rel_length = (completion_length - mean) / std
                penalty = sigmoid(rel_length)
                score = 1.0 * (1 - self.alpha * penalty)
            else:
                score = 0.0
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][data_item.batch['prompts'].shape[-1]:].sum()
            reward_tensor[i, valid_response_length - 1] = score
            data_source = data_item.non_tensor_batch['data_source']
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(response_ids[:valid_response_length], skip_special_tokens=True)
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", score)
        return reward_tensor 