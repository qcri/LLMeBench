import os
import random
import re

from llmebench.datasets import CT22CheckworthinessDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import CheckworthinessTask


random.seed(1333)


def config():
    return {
        "dataset": CT22CheckworthinessDataset,
        "dataset_args": {},
        "task": CheckworthinessTask,
        "task_args": {},
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["0", "1"],
            "max_tries": 30,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/checkworthyness/arabic/CT22_arabic_1A_checkworthy_test_gold.tsv"
        },
    }


def prompt(input_sample):
    prompt_string = (
        f'Annotate the "tweet" into "one" of the following categories: checkworthy or not_checkworthy\n\n'
        f"tweet: {input_sample}\n"
        f"label: \n"
    )
    return [
        {
            "role": "system",
            "content": "As an AI system, your role is to analyze tweets and classify them as 'checkworthy' or 'not_checkworthy' based on their potential importance for journalists and fact-checkers.",
        },
        {
            "role": "user",
            "content": prompt_string,
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"]

    label = label.replace("label:", "").strip()

    if "label: " in label:
        arr = label.split("label: ")
        label = arr[1].strip()

    if label == "checkworthy" or label == "Checkworthy":
        label_fixed = "1"
    elif label == "Not_checkworthy." or label == "not_checkworthy":
        label_fixed = "0"
    elif "not_checkworthy" in label or "label: not_checkworthy" in label:
        label_fixed = "0"
    elif "checkworthy" in label or "label: checkworthy" in label:
        label_fixed = "1"
    else:
        label_fixed = None

    return label_fixed
