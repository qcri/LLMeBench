import os
import random
import re

from arabic_llm_benchmark.datasets import FactualityKhouja20Dataset
from arabic_llm_benchmark.models import GPTChatCompletionModel
from arabic_llm_benchmark.tasks import FactualityKhouja20Task


random.seed(1333)


def config():
    return {
        "dataset": FactualityKhouja20Dataset,
        "dataset_args": {},
        "task": FactualityKhouja20Task,
        "task_args": {},
        "model": GPTChatCompletionModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": "gpt-4-32k",
            "class_labels": ["true", "false"],
            "max_tries": 30,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/factuality_stance_khouja/claim/test.csv"
        },
    }


def prompt(input_sample):
    prompt_string = (
        f'Annotate the "tweet" into one of the following categories: correct or incorrect\n\n'
        f"tweet: {input_sample}\n"
        f"label: \n"
    )
    return [
        {
            "role": "system",
            "content": "You are a social media expert and a fact-checker.",  # You are capable of identifying and annotating tweets correct or incorrect
        },
        {
            "role": "user",
            "content": prompt_string,
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"].lower()

    if (
        label.startswith("I am unable to verify".lower())
        or label.startswith("I am unable to categorize".lower())
        or label.startswith(
            "I am an AI language model and I am unable to verify".lower()
        )
        or label.startswith("I am an AI language model, and I am unable".lower())
        or label.startswith("I cannot categorize this".lower())
    ):
        label_fixed = None
    elif "label: incorrect" in label or "incorrect" in label:
        label_fixed = "false"
    elif "label: correct" in label or "correct" in label:
        label_fixed = "true"
    else:
        label_fixed = None

    return label_fixed
