import yaml
import json
import requests
from datasets import *

with open("config.yml", 'r') as file:
    config = yaml.safe_load(file)


def sample(prompt, length_penalty=0.7, temperature=0.9, max_tokens=200, stop=[], seed=1234, max_tries=5):
    tries = 0
    while True:
        try:
            tries += 1
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
                    "seed": seed,
                    "sensitive_check": False
                }),
                timeout=60
            )
            response.raise_for_status()
        except Exception as e:
            print(e)
            continue

        if response.status_code != 204:
            response_json = response.json()
            if response_json['status'] == 0 and 'text' in response_json['result']['output']:
                return response_json['result']['output']['text']
        else:
            print("204: No content returned by server")

        if tries >= max_tries:
            raise Exception(f"Server could not generate text after {max_tries} tries")
