import os
import random
import re

from arabic_llm_benchmark.datasets import CheckworthinessDataset
from arabic_llm_benchmark.models import GPTChatCompletionModel
from arabic_llm_benchmark.tasks import CheckworthinessTask


random.seed(1333)


def config():
    return {
        "dataset": CheckworthinessDataset,
        "dataset_args": {},
        "task": CheckworthinessTask,
        "task_args": {},
        "model": GPTChatCompletionModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": "gpt-4-32k",
            "class_labels": ["0", "1"],
            "max_tries": 30,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/harmful/CT22_arabic_1C_harmful_test_gold.tsv"
        },
    }


def prompt(input_sample):
    prompt_string = (
        f'Given the following tweet, analyze its content and identify if it could potentially cause harm to an individual, society, or an organization. Annotate the tweet as "harmful" if it has the potential to cause harm, or "not_harmful" if it does not.\n\n'
        f"tweet: {input_sample}\n"
        f"label: \n"
    )
    return [
        {
            "role": "system",
            "content": "As a system, you are an AI-based social media expert. You are capable of identifying and annotating tweets that could potentially cause harm to individuals, society, or organizations. Your role is to analyze the content of each tweet provided to you, assess its potential impact, and categorize it as harmful or not based on your advanced understanding of harmful speech patterns, social contexts, and potential implications.",
        },
        {
            "role": "user",
            "content": prompt_string,
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"]

    if "label: harmful" in label:
        label_fixed = "1"
    elif "label: not_harmful" in label:
        label_fixed = "0"
    elif "label: " in label:
        arr = label.split("label: ")
        label = arr[1].strip()

    elif label.startswith("harmful") or label.startswith("yes"):
        label_fixed = "1"

    elif (
        label.startswith("no")
        or label == "label: safe"
        or label == "not_harmful"
        or "not harmful" in label
    ):
        label_fixed = "0"
    else:
        label = label.replace(".", "").strip().lower()
        label = label.replace("label:", "").strip()
        label_fixed = label

    return label_fixed
