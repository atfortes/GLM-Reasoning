import re
import sys
import yaml
import json
import wandb
import argparse
import datasets
import traceback
from inference import sample


def extract_answer(question, prediction, ans_re):
    answer = re.findall(ans_re, prediction.lower().partition("the answer is ")[2] if wandb.config.type == 'f-cot' else prediction)
    if len(answer) > 0:
        return answer[0]
    elif wandb.config.dataset in ("CommonsenseQA",):
        part = question.partition(prediction)
        if part[1] != "":
            answer = re.findall(ans_re, part[0].split("\n")[-1])
        if len(answer) > 0:
            return answer[0]
    return "[NO ANSWER]"


def evaluate():
    wandb.init()
    total_correct = 0
    total_correct_voters = 0
    all_predictions_log_data = []

    with open("config.yml", 'r') as file:
        config = yaml.safe_load(file)

    dataset = getattr(datasets, wandb.config.dataset)(config, "test.jsonl")

    if wandb.config.type == 'z':
        prefix = ""
        answer_prefix = dataset.ZERO_SHOT_PROMPT
    if wandb.config.type == 'z-cot':
        prefix = ""
        answer_prefix = "Let's think step by step."
    elif wandb.config.type == 'f':
        prefix = "\n\n".join(json.load(open(
                 "few-shot-exemplars/few-shot-exemplars.json", 'r'))[wandb.config.dataset]) + "\n\n"
        answer_prefix = "The answer is [MASK]."
    elif wandb.config.type == 'f-cot':
        prefix = "\n\n".join(json.load(open(
                 "few-shot-exemplars/few-shot-cot-exemplars.json", 'r'))[wandb.config.dataset]) + "\n\n"
        answer_prefix = ""

    samples_num = min(len(dataset.questions), wandb.config.samples_num)
    for i in range(1, samples_num + 1):
        try:
            predictions_log_data = []

            question = dataset.questions[i]
            prompt = f'''{prefix}Q: {question}\nA: {answer_prefix}'''

            target = dataset.answers[i]
            predictions_full = [sample(
                prompt,
                wandb.config.length_penalty,
                wandb.config.temperature,
                wandb.config.max_tokens,
                wandb.config.stop_token,
                seed
            )[0] for seed in range(1, wandb.config.voting_num + 1)]

            if wandb.config.type == 'z-cot':
                predictions_full = [sample(
                    f"{prompt} {chain_of_thought} The answer is [MASK].",
                    wandb.config.length_penalty,
                    wandb.config.temperature,
                    10
                )[0] for chain_of_thought in predictions_full]

            predictions = [extract_answer(question, p, dataset.ANS_RE)
                           for p in predictions_full]
            predictions_log_data = [[question, p_full, p, target]
                                    for p_full, p in zip(predictions_full, predictions)]
            all_predictions_log_data.extend(predictions_log_data)
            score = sum([int(p == target) for p in predictions]) / wandb.config.voting_num
            total_correct_voters += sum([int(p == target) for p in predictions])
            final_prediction = max(set(predictions), key=predictions.count)
            correct = final_prediction == target
            total_correct += correct

            wandb.log({
                'correct': int(correct),
                'correct ratio': score,
                'accuracy': total_correct / i,
                'predictions': wandb.Table(
                    data=predictions_log_data,
                    columns=["question", "prediction (full)", "prediction (extracted)", "target"])
            })
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"error in {i}: {e}")
            break

    wandb.log({
        'total accuracy': total_correct / samples_num,
        'total correct ratio': total_correct_voters / (samples_num * wandb.config.voting_num),
        'all_predictions': wandb.Table(
            data=all_predictions_log_data,
            columns=["question", "prediction (full)", "prediction (extracted)", "target"]
        )
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', '--dataset', type=str, default="GSM8K",
                        choices=["GSM8K", "CommonsenseQA", "StrategyQA"])
    parser.add_argument('-t', '--type', type=str, default="z-cot",
                        choices=["z", "z-cot", "f", "f-cot"])
    parser.add_argument('-sn', '--samples_num', type=int, default=2000)
    parser.add_argument('-vn', '--voting_num', type=int, default=1)
    parser.add_argument('-lp', '--length_penalty', type=float, default=0.99)
    parser.add_argument('-tmp', '--temperature', type=float, default=0.01)
    parser.add_argument('-mt', '--max_tokens', type=int, default=128)
    parser.add_argument('-st', '--stop_token', type=list, default=["Q:"])
    args = parser.parse_args()

    sweep_configuration = {
        'method': 'random',
        'name': 'GSM8K (zero-shot CoT)',
        'metric': {'goal': 'maximize', 'name': 'accuracy'},
        'parameters':
        {
            'dataset': {'value': args.dataset},
            'type': {'value': args.type},
            'samples_num': {'value': args.samples_num},
            'max_tokens': {'value': args.max_tokens},
            'stop_token': {'value': args.stop_token},
            'voting_num': {'value': args.voting_num},
            'length_penalty': {'value': args.length_penalty},
            'temperature': {'value': args.temperature}
        }
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project='GLM-Reasoning')
    wandb.agent(sweep_id, function=evaluate, count=1)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        # exit gracefully, so wandb logs the problem
        print(traceback.print_exc(), file=sys.stderr)
        exit(1)
