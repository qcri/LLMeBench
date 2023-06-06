import os

from arabic_llm_benchmark.datasets import CheckworthinessDataset
from arabic_llm_benchmark.models import GPTModel, RandomGPTModel
from arabic_llm_benchmark.tasks import CheckworthinessTask


def config():
    return {
        "dataset": CheckworthinessDataset,
        "dataset_args": {},
        "task": CheckworthinessTask,
        "task_args": {},
        "model": RandomGPTModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": "gpt",
            "class_labels": [
                "yes_discusses_action_taken",
                "harmful",
                "yes_discusses_cure",
                "yes_asks_question",
                "no_not_interesting",
                "yes_other",
                "yes_blame_authorities",
                "yes_contains_advice",
                "yes_calls_for_action",
            ],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/checkworthyness/CT22_arabic_1A_checkworthy_test_gold.tsv"
        },
    }


def prompt(input_sample):
    ## GPT 3.5 - turbo
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": f'Classify the "tweet" as checkworthy or not_checkworthy. Provide only label.\n\nsentence: {input_sample}label: ',
            }
        ],
    }


def post_process(response):
    # label = response["choices"][0]['text']
    label = response["response"]["choices"][0]["message"]["content"]

    if label == "checkworthy":
        label_fixed = "1"
    elif label == "Not_checkworthy." or label == "not_checkworthy":
        label_fixed = "0"

    return label_fixed
