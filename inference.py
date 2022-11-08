import requests
import json
from datasets import GSM8K
import re
import yaml

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

gsm8k = GSM8K(config)

# Construct prompt
prompt = "Answer the following questions. Show your work and write your answer after '####'. \n"
prompt += "".join(["Question: " + example['question'] + "\n" +
                  "Answer: " + example['answer'] + "\n" for example in gsm8k.train[:5]])


def sample(question, seed):

    full_prompt = prompt + "Question: " + question + \
        " Answer: Let's think step by step. "
    
    # Remove calculator tokens from dataset
    full_prompt = re.sub(r'{}.*?{}'.format(re.escape("<<"),
                                           re.escape(">>")), '', full_prompt)

    headers = {'Content-Type': 'application/json'}
    data = {
        "apikey": config['api_key'],
        "apisecret": config['api_secret'],
        "language": "en-US",
        "prompt": full_prompt,
        "length_penalty": 0.7,
        "sampling_strategy": "BaseStrategy",
        "temperature": 0.5,
        "top_k": 40,
        "top_p": 0,
        "min_gen_length": 10,
        "max_tokens": 512,
        "seed": seed
    }

    response = requests.post(
        config['request_url'], headers=headers, data=json.dumps(data))
    return response.json()['result']['output']['text']


for i in range(1, 5):
    print(sample(gsm8k.train[5]['question'], seed=i))
