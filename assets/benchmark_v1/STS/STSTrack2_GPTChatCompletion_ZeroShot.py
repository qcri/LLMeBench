import os
import re

from arabic_llm_benchmark.datasets import STSTrack2Dataset
from arabic_llm_benchmark.models import GPTChatCompletionModel
from arabic_llm_benchmark.tasks import STSTrack2Task


def config():
    return {
        "dataset": STSTrack2Dataset,
        "dataset_args": {},
        "task": STSTrack2Task,
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
            "train_data_path": "",
            "n_shots": 3
        },
    }


def prompt(input_sample,examples):
    prompt_string = f"Given two sentences, produce a continuous valued similarity score on a " \
                    f"scale from 0 to 5, with 0 indicating that the semantics of the sentences are " \
                    f"completely independent and 5 signifying semantic equivalence. The output " \
                    f"should be exactly in form Similarity score =. \n{input_sample}"
    return [
        {
            "role": "system",
            "content": "You are an AI assistant that helps people find information.",
        },
        {
            "role": "user",
            "content": prompt_string,
        },
    ]


def post_process(response):
    raw_response = response["choices"][0]["message"]["content"]

    if "Similarity score =" in raw_response:
        pred_num = raw_response.split("Similarity score = ")[1].strip().split(" ")[0].rstrip(".")
        score = float(pred_num)

    else:
        score = None

    return score
