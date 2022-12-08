import re
import sys
import yaml
import json
import wandb
import argparse
import datasets
import traceback
from inference import sample


def evaluate():
    wandb.init()

    total_correct = 0
    total_correct_voters = 0
    log_data = []

    with open("config.yml", 'r') as file:
        config = yaml.safe_load(file)

    dataset = getattr(datasets, wandb.config.dataset)(config, "dev_rand_split.jsonl")

    if wandb.config.type == 'z':
        prefix = ""
        answer_prefix = dataset.ZERO_SHOT_PROMPT
    elif wandb.config.type == 'f':
        prefix = "\n\n".join(json.load(open(
                 "few-shot-exemplars/few-shot-exemplars.json", 'r'))[wandb.config.dataset]) + "\n\n"
        answer_prefix = ""
    elif wandb.config.type == 'f-cot':
        prefix = "\n\n".join(json.load(open(
                 "few-shot-exemplars/few-shot-cot-exemplars.json", 'r'))[wandb.config.dataset]) + "\n\n"
        answer_prefix = ""

    answer_partition = "answer is"

    samples_num = min(len(dataset.questions), wandb.config.samples_num)
    for i in range(1, samples_num + 1):
        try:
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

            predictions = [re.findall(dataset.ANS_RE, p if wandb.config.type == 'z' else p.partition(answer_partition)[2])
                        for p in predictions_full]
            predictions = [p[0] if len(p) > 0 else "[NO ANSWER]" for p in predictions]
            log_data.extend([[question, p_full, p, target]
                            for p_full, p in zip(predictions_full, predictions)])
            score = sum([int(p == target) for p in predictions]) / wandb.config.voting_num
            total_correct_voters += sum([int(p == target) for p in predictions])
            final_prediction = max(set(predictions), key=predictions.count)
            correct = final_prediction == target
            total_correct += correct

            wandb.log({
                'correct': int(correct),
                'correct ratio': score,
                'accuracy': total_correct / i
            })
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"error in {i}: {e}")
            break

    wandb.log({
        'total accuracy': total_correct / samples_num,
        'total correct ratio': total_correct_voters / (samples_num * wandb.config.voting_num),
        'predictions': wandb.Table(
            data=log_data,
            columns=["question", "prediction (full)", "prediction (extracted)", "target"]),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', '--dataset', type=str, default="CommonsenseQA",
                        choices=["GSM8K", "CommonsenseQA", "StrategyQA"])
    parser.add_argument('-t', '--type', type=str, default="f",
                        choices=["z", "f", "f-cot"])
    parser.add_argument('-sn', '--samples_num', type=int, default=2000)
    parser.add_argument('-vn', '--voting_num', type=int, default=1)
    parser.add_argument('-lp', '--length_penalty', type=float, default=0.99)
    parser.add_argument('-tmp', '--temperature', type=float, default=0.01)
    parser.add_argument('-mt', '--max_tokens', type=int, default=10)
    parser.add_argument('-st', '--stop_token', type=list, default=["Q:"])
    args = parser.parse_args()

    sweep_configuration = {
        'method': 'random',
        'name': 'CommonsenseQA (zero-shot)',
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
