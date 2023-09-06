import os

from llmebench.datasets import OSACT4SubtaskADataset
from llmebench.models import LegacyOpenAIModel
from llmebench.tasks import OffensiveTask


def config():
    return {
        "dataset": OSACT4SubtaskADataset,
        "dataset_args": {},
        "task": OffensiveTask,
        "task_args": {},
        "model": LegacyOpenAIModel,
        "model_args": {
            "class_labels": ["OFF", "NOT_OFF"],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/offensive_language/OSACT2020-sharedTask-test-tweets-labels.txt"
        },
    }


def prompt(input_sample):
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": f'if the following Arabic sentence is offensive, just say "OFF", otherwise, say just "NOT_OFF" without explanation: \n {input_sample}',
            }
        ],
    }


def post_process(response):
    out = response["choices"][0]["text"]
    j = out.find(".")
    if j > 0:
        out = out[0:j]
    return out
