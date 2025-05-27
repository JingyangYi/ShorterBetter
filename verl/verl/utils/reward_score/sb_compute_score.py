

def sb_compute_score(data_source, solution_str, ground_truth, optimal_length, completion_length, correct_or_not, extra_info=None):
    """return the reward score based on sb func; optimal length is the shortest length among all correct completions"""
    if correct_or_not == True:
        correctness = 1.0
    else:
        correctness = 0.0
    length_gap = abs(completion_length - optimal_length) * 0.001

    return correctness - length_gap