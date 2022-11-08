import json
import re

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_completion, gt_example):
    gt_answer = extract_answer(gt_example["answer"])
    assert gt_answer != INVALID_ANS
    return extract_answer(model_completion) == gt_answer


class GSM8K():

    def __init__(self, config):
        with open(config['GSM8K_path'] + '/train.jsonl') as f:
            self.train = [json.loads(l) for l in f.readlines()]

        with open(config['GSM8K_path'] + '/test.jsonl') as f:
            self.test = [json.loads(l) for l in f.readlines()]
