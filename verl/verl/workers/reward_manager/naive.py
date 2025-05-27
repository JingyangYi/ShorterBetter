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
from verl.utils.reward_score import _default_compute_score
import torch


class NaiveRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, train_batch_size, num_generation, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.train_batch_size = train_batch_size
        self.num_generation = num_generation

    def verify(self, data):
        scores = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            scores.append(score)
        data.batch['acc'] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores
    
    # def check_correctness_and_length(self, data):
    #     """in GRPO, the input data here should have len=train_batch_size * num_generations(i.e. config.actor_rollout_ref.rollout.n)"""
    #     train_batch_size = self.train_batch_size
    #     num_generation = self.num_generation

    #     print(data)
    #     assert len(data) == train_batch_size * num_generation, f"Implementation Error: the input data should have len={train_batch_size * num_generation}, where train_batch_size={train_batch_size}, num_generation={num_generation}, while len(data)={len(data)}"

    #     start_points = list(range(0, len(data), num_generation))

    #     correctness_set = []
    #     length_set = []
    #     optimal_length_set = []
    #     for start_point in range(len(start_points)):
    #         completions_given_prompt = data[start_point:start_point + num_generation]

    #         prompts = []  # they should be the same
    #         response_lengths_given_prompt = []
    #         correctnesses_given_prompt = []
    #         for completion in completions_given_prompt:
    #             prompt_ids = completion.batch['prompts']

    #             prompt_length = prompt_ids.shape[-1]

    #             valid_prompt_length = completion.batch['attention_mask'][:prompt_length].sum()
    #             valid_prompt_ids = prompt_ids[-valid_prompt_length:]

    #             response_ids = completion.batch['responses']
    #             valid_response_length = completion.batch['attention_mask'][prompt_length:].sum()
    #             valid_response_ids = response_ids[:valid_response_length]

    #             # decode
    #             prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
    #             response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

    #             ground_truth = completion.non_tensor_batch['reward_model']['ground_truth']

    #             data_source = completion.non_tensor_batch['data_source']

    #             extra_info = completion.non_tensor_batch.get('extra_info', None)

    #             # check correctness
    #             correct_or_not = verify_correctness(
    #                 solution_str=response_str,
    #                 ground_truth=ground_truth,
    #             )
    #             correctness_set.append(correct_or_not)
    #             correctnesses_given_prompt.append(correct_or_not)
    #             length_set.append(valid_response_length)
    #             response_lengths_given_prompt.append(valid_response_length)

    #             prompts.append(prompt_str)

    #         # the shortest correct response length is selected as optimal length
    #         correct_lengths = [l for c, l in zip(correctnesses_given_prompt, response_lengths_given_prompt) if c]
    #         optimal_length = min(correct_lengths) if correct_lengths else sum(response_lengths_given_prompt)/len(response_lengths_given_prompt)
    #         optimal_length_set.extend([optimal_length] * num_generation)

    #         assert len(set(prompts)) == 1, f"Implementation Error: in this given dataproto prompts should be the same, while prompts={prompts}"
    #     assert len(correctness_set) == len(length_set) == optimal_length_set == len(data), f"Implementation Error: the length of correctness_set={len(correctness_set)}, length_set={len(length_set)}, optimal_length_set={len(optimal_length_set)}, data={len(data)}, they should all be the same length"

    #     return correctness_set, length_set, optimal_length_set

    def check_correctness_and_length(self, data):
        """in GRPO, the input data here should have len=train_batch_size * num_generations(i.e. config.actor_rollout_ref.rollout.n)"""
        train_batch_size = self.train_batch_size
        num_generation = self.num_generation
        
        assert len(data) == train_batch_size * num_generation, f"Implementation Error: the input data should have len={train_batch_size * num_generation}, where train_batch_size={train_batch_size}, num_generation={num_generation}, while len(data)={len(data)}"
        
        # Extract unique problem IDs to group responses
        problem_ids = [item.non_tensor_batch['index'] for item in data]
        unique_problem_ids = []
        for pid in problem_ids:
            if pid not in unique_problem_ids:
                unique_problem_ids.append(pid)
        
        # Should have train_batch_size unique problems
        assert len(unique_problem_ids) == train_batch_size, f"Expected {train_batch_size} unique problems, got {len(unique_problem_ids)}"
        
        # Create a mapping from data index to results
        correctness_map = {}
        length_map = {}
        optimal_length_map = {}
        
        # Process each unique problem
        for problem_id in unique_problem_ids:
            # Find all instances of this problem in the batch
            indices = [i for i in range(len(data)) if data[i].non_tensor_batch['index'] == problem_id]
            completions_given_prompt = [data[i] for i in indices]
            
            # Should have num_generation repetitions of each problem
            assert len(completions_given_prompt) == num_generation, f"Expected {num_generation} repetitions for problem {problem_id}, got {len(completions_given_prompt)}"
            
            prompts = []  # they should be the same
            response_lengths_given_prompt = []
            correctnesses_given_prompt = []
            
            for idx, completion in zip(indices, completions_given_prompt):
                prompt_ids = completion.batch['prompts']
                
                prompt_length = prompt_ids.shape[-1]
                
                valid_prompt_length = completion.batch['attention_mask'][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]
                
                response_ids = completion.batch['responses']
                valid_response_length = completion.batch['attention_mask'][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]
                
                # decode
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                
                ground_truth = completion.non_tensor_batch['reward_model']['ground_truth']
                
                # check correctness
                correct_or_not = verify_correctness(
                    solution_str=response_str,
                    ground_truth=ground_truth,
                )
                correctnesses_given_prompt.append(correct_or_not)
                response_lengths_given_prompt.append(valid_response_length)
                
                # Store in the map
                correctness_map[idx] = correct_or_not
                length_map[idx] = valid_response_length
                
                prompts.append(prompt_str)
            
            # the shortest correct response length is selected as optimal length
            correct_lengths = [l for c, l in zip(correctnesses_given_prompt, response_lengths_given_prompt) if c]


            # --------------------------------------------------
            # TEST: WHAT IF JUST CHOOSE THE MIN LENGTH, REGRARDLESS OF CORRECTNESS?
            optimal_length = min(correct_lengths) if correct_lengths else sum(response_lengths_given_prompt)/len(response_lengths_given_prompt)
            # optimal_length = min(response_lengths_given_prompt)
            # --------------------------------------------------



            # Just print a compact summary line for each problem
            correct_count = sum(correctnesses_given_prompt)
            print(f"Lengths={response_lengths_given_prompt}, Correct={correct_lengths}", flush=True)
            
            # Store optimal length for all instances of this problem
            for idx in indices:
                optimal_length_map[idx] = optimal_length
            
            assert len(set(prompts)) == 1, f"Implementation Error: in this given dataproto prompts should be the same, while prompts={prompts}"
        
        
        # Convert maps back to lists in the original order
        correctness_set = [correctness_map[i] for i in range(len(data))]
        length_set = [length_map[i] for i in range(len(data))]
        optimal_length_set = [optimal_length_map[i] for i in range(len(data))]
        
        assert len(correctness_set) == len(length_set) == len(optimal_length_set) == len(data), f"Implementation Error: the length of correctness_set={len(correctness_set)}, length_set={len(length_set)}, optimal_length_set={len(optimal_length_set)}, data={len(data)}, they should all be the same length"
        
        return correctness_set, length_set, optimal_length_set
    
    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        correctness_set, length_set, optimal_length_set = self.check_correctness_and_length(data)

        def sb_compute_score(optimal_length, completion_length, correct_or_not):
            """return the reward score based on sb func; optimal length is the shortest length among all correct completions"""
            alpha = 2.0
            beta = 0.001
            if correct_or_not == True:
                correctness = alpha
            else:
                correctness = 0.0
            length_gap = abs(completion_length - optimal_length) * beta

            return correctness - length_gap

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            
            correct_or_not = correctness_set[i]
            completion_length = length_set[i]
            optimal_length = optimal_length_set[i]

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = sb_compute_score(optimal_length, completion_length, correct_or_not)
            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", score)

        return reward_tensor




# Below are the helper functions to verify the mathematical equivalence of two strings
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


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
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


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
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


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string
