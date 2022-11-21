import wandb
import argparse
import json
import yaml
import datasets
import re
import traceback
import sys

import numpy as np
from inference import sample


def evaluate():
    run = wandb.init()

    total_correct = 0
    total_correct_members = 0
    log_data = []

    with open("config.yml", 'r') as file:
        config = yaml.safe_load(file)

    if wandb.config.language == 'en-US':
        few_shot_cot_exemplars = "\n\n".join(json.load(open(
            "few-shot-exemplars/few-shot-cot-exemplars.json", 'r'))[wandb.config.dataset])
        file_name = "train.jsonl"
        question_marker = "Q"
        answer_marker = "A"

        answer_prefix = "The answer is"

    elif wandb.config.language == 'zh-CN':
        few_shot_cot_exemplars = "\n\n".join(json.load(open(
            "few-shot-exemplars/few-shot-cot-exemplars-cn.json", 'r'))[wandb.config.dataset])
        file_name = "train_cn.jsonl"
        question_marker = "问"
        answer_marker = "答"

        answer_prefix = "答案是"

    else:
        raise ValueError("language needs to be en-US or zh-CN!")

    dataset = getattr(datasets, wandb.config.dataset)(config, file_name)

    samples_num = min(len(dataset.questions), wandb.config.samples_num)
    for i in np.random.choice(range(len(dataset.questions)), size=samples_num) if wandb.config.random_samples else range(samples_num):
        question = dataset.questions[i]
        prompt = f'''{few_shot_cot_exemplars}\n\n{question_marker}: {question}\n{answer_marker}:'''

        target = dataset.answers[i]
        predictions_full = [sample(
            prompt,
            wandb.config.length_penalty,
            wandb.config.temperature,
            wandb.config.max_tokens,
            wandb.config.stop_token,
            seed
        )[0] for seed in range(1, wandb.config.voting_num)]

        predictions = [re.findall(
            r'\d+', p.partition(answer_prefix)[2]) for p in predictions_full]
        predictions = [p[0] if len(p) > 0 else "" for p in predictions]

        log_data.extend([[question, p_full, p, target]
                        for p_full, p in zip(predictions_full, predictions)])

        score = sum([int(p == target)
                    for p in predictions]) / wandb.config.voting_num
        total_correct_members += sum([int(p == target) for p in predictions])
        final_prediction = max(set(predictions), key=predictions.count)
        correct = int(final_prediction == target)
        total_correct += correct

        wandb.log({
            'correct ratio': score,
            'correct': correct
        })

    columns = ["question",
               "prediction (full)", "prediction (extracted)", "target"]
    wandb.log({
        'val_acc': total_correct / samples_num,
        'val_correct_ratio': total_correct_members / (samples_num * wandb.config.voting_num),
        'predictions': wandb.Table(data=log_data, columns=columns),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', '--dataset', type=str, default="GSM8K",
                        choices=["GSM8K", "CommonsenseQA", "StrategyQA"])
    parser.add_argument('-sn', '--samples_num', type=int, default=10)
    parser.add_argument('-rs', '--random_samples', type=bool, default=False)
    parser.add_argument('-lp', '--length_penalty', type=float, default=0.7)
    parser.add_argument('-t', '--temperature', type=float, default=0.1)
    parser.add_argument('-mt', '--max_tokens', type=int, default=128)
    parser.add_argument('-st', '--stop_token', type=list, default=["\n\n"])
    parser.add_argument('-vn', '--voting_num', type=int, default=1)
    args = parser.parse_args()

    sweep_configuration = {
        'method': 'bayes',
        'name': 'GSM8K_parameter_sweep',
        'metric': {'goal': 'maximize', 'name': 'correct ratio'},
        'parameters':
        {
            'dataset': {'value': args.dataset},
            'samples_num': {'value': args.samples_num},
            'random_samples': {'value': args.random_samples},
            'max_tokens': {'value': args.max_tokens},
            'stop_token': {'value': args.stop_token},
            'voting_num': {'value': args.voting_num},

            'language': {'values': ['en-US', 'zh-CN']},
            'length_penalty': {'max': 0.99, 'min': 0.01},
            'temperature': {'max': 0.99, 'min': 0.01}
        }
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project='GLM-Reasoning')
    wandb.agent(sweep_id, function=evaluate, count=30)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        # exit gracefully, so wandb logs the problem
        print(traceback.print_exc(), file=sys.stderr)
        exit(1)
