import os

from llmebench.datasets import AdultDataset
from llmebench.models import LegacyOpenAIModel
from llmebench.tasks import AdultTask


def config():
    return {
        "dataset": AdultDataset,
        "dataset_args": {},
        "task": AdultTask,
        "task_args": {},
        "model": LegacyOpenAIModel,
        "model_args": {
            "class_labels": ["ADULT", "NOT_ADULT"],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/adult/adult-test.tsv"
        },
    }


def prompt(input_sample):
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": f'Classify the following Arabic sentence as adult language (the language used in adult advertisement and porno advertisement) or not adult language without illustruation. In case of adult language, just write "ADULT" without explaination, and in case of not adult language, just write "NOT_ADULT" without explaination \n {input_sample}',
            }
        ],
    }


def post_process(response):
    out = response["choices"][0]["text"]
    j = out.find(".")
    if j > 0:
        out = out[0:j]
    return out
