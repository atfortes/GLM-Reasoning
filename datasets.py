import json
import re


class GSM8K:
    # math word problems

    ZERO_SHOT_PROMPT = "The answer (numeric) is [MASK]."
    ANS_RE = re.compile(r"(\-?[0-9\.\,]+)")

    def __init__(self, config, file_name="test.jsonl"):
        with open(config["GSM8K_path"] + file_name) as f:
            samples = list(map(json.loads, f.readlines()))
            self.questions = [s["question"] for s in samples]
            self.answers = [self.extract_answer(s["answer"]) for s in samples]
        assert len(self.questions) == len(self.answers)

    def extract_answer(self, completion):
        match = re.compile(r"#### (\-?[0-9\.\,]+)").search(completion)
        match_str = match.group(1).strip()
        return match_str


class CommonsenseQA:
    # multiple choice questions

    ZERO_SHOT_PROMPT = "Among (a) through (e), the answer is [MASK]."
    ANS_RE = re.compile(r"\(a\)|\(b\)|\(c\)|\(d\)|\(e\)")

    def __init__(self, config, file_name="dev_rand_split.jsonl"):
        with open(config["CommonsenseQA_path"] + file_name) as f:
            samples = list(map(json.loads, f.readlines()))
            self.questions = list(map(
                lambda s: s["question"]["stem"] + "\nAnswer Choices:\n" + \
                    "\n".join(map(lambda c: f'''({c["label"].lower()}) {c["text"]}''', s["question"]["choices"])), samples
            ))
            self.answers = list(map(lambda s: f'''({s["answerKey"].lower()})''', samples))
        assert len(self.questions) == len(self.answers)


class StrategyQA:
    # yes or no questions

    ZERO_SHOT_PROMPT = "The answer (yes or no) is [MASK]."
    ANS_RE = re.compile(r"yes|no")

    def __init__(self, config, file_name="strategyqa_train.json"):
        with open(config["StrategyQA_path"] + file_name, encoding='utf-8') as f:
            samples = json.load(f)
            self.questions = list(map(lambda s: s["question"], samples))
            self.answers = list(map(lambda s: "yes" if s["answer"] else "no", samples))
        assert len(self.questions) == len(self.answers)
