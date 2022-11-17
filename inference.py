import yaml
import json
import requests
from datasets import *

with open("config.yml", 'r') as file:
    config = yaml.safe_load(file)


def sample(prompt, length_penalty=0.7, temperature=0.9, max_tokens=200, stop=[], seed=1234):
    response = requests.post(
        config['request_url'],
        headers={"Content-Type": "application/json"},
        data=json.dumps({
            "apikey": config["api_key"],
            "apisecret": config["api_secret"],
            "language": "en-US",
            "prompt": prompt,
            "length_penalty": length_penalty,
            "sampling_strategy": "BaseStrategy",
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": stop,
            "seed": seed
        })
    )
    return response.json()['result']['output']['text']
