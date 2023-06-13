import os
import re

from arabic_llm_benchmark.datasets import STSTrack1Dataset
from arabic_llm_benchmark.models import GPTChatCompletionModel
from arabic_llm_benchmark.tasks import STSTrack1Task


def config():
    return {
        "dataset": STSTrack1Dataset,
        "dataset_args": {},
        "task": STSTrack1Task,
        "task_args": {},
        "model": GPTChatCompletionModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": "gpt-4-32k",
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/STS/semeval-2017",
            'train_data_path' : "none",
            'n_shots' : 2
            # "ground_truth_data_path": "data/STS/semeval-2017/STS2017.eval.v1.1/STS.input.track1.ar-ar.txt"
        },
    }


def prompt(input_sample, examples):
    sent1, sent2 = input_sample.split("\t")
    prompt_string = (
        f'Given sentence1 and sentence2, compute semantic similarity between them using a score on a scale from 0.0 '
        f'to 5.0, with 0.0 indicating least similarity between the sentences, and 5.0 signifying very strong semantic '
        f'similarity. The output should be exactly in form score=\n\n'
        f"sentence1: {sent1}\n"
        f"sentence2: {sent2}\n"
        f"score= \n"
    )
    return [
        {
            "role": "system",
            "content": "As an AI system, your role is to compute semantic similarity between two Arabic sentences.",
        },
        {
            "role": "user",
            "content": prompt_string,
        },
    ]


def post_process(response):
    raw_response = response["choices"][0]["message"]["content"].lower().strip()
    regex_float = r"\b\d+\.\d+\.?\b"

    if "score=" in raw_response:
        match = re.findall(regex_float, raw_response)[0]
        score = float(match)
    else:
        print(raw_response)
        score = None

    return score
