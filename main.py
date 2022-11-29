import yaml
import json
import datasets
import argparse
import numpy as np
from inference import sample


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', '--dataset', type=str, default="GSM8K", choices=["GSM8K", "CommonsenseQA", "StrategyQA"])
    parser.add_argument('-sn', '--samples_num', type=int, default=10)
    parser.add_argument('-rs', '--random_samples', type=bool, default=False)
    parser.add_argument('-lp', '--length_penalty', type=float, default=0.7)
    parser.add_argument('-t', '--temperature', type=float, default=0.1)
    parser.add_argument('-mt', '--max_tokens', type=int, default=128)
    parser.add_argument('-st', '--stop_token', type=list, default=["#"])
    parser.add_argument('-s', '--seed', type=int, default=1234)
    args = parser.parse_args()

    with open("config.yml", 'r') as file:
        config = yaml.safe_load(file)

    print('Loading dataset...')
    dataset = getattr(datasets, args.dataset)(config)
    few_shot_cot_exemplars = "#\n" + "\n#\n".join(json.load(open("few-shot-exemplars/few-shot-cot-exemplars.json", 'r'))[args.dataset])

    print('Sampling...')
    samples_num = min(len(dataset.questions), args.samples_num)
    for i in np.random.choice(range(len(dataset.questions)), size=samples_num) if args.random_samples else range(samples_num):
        question = dataset.questions[i]
        prompt = f'''{few_shot_cot_exemplars}\n#\nQ: {question}\nA:'''
        print("="*34, "Question", i, "="*34)
        print(question)
        print("Correct Answer:", dataset.answers[i])
        print("Predicted Answer:", sample(
            prompt,
            args.length_penalty,
            args.temperature,
            args.max_tokens,
            args.stop_token,
            args.seed
        ))
