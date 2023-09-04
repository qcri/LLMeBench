import os

from llmebench.datasets import OSACT4SubtaskADataset
from llmebench.models import BLOOMPetalModel
from llmebench.tasks import OffensiveTask


def config():
    return {
        "dataset": OSACT4SubtaskADataset,
        "dataset_args": {},
        "task": OffensiveTask,
        "task_args": {},
        "model": BLOOMPetalModel,
        "model_args": {
            "api_url": os.environ["API_URL"],
            "class_labels": ["OFF", "NOT_OFF"],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/offensive_language/OSACT2020-sharedTask-test-tweets-labels.txt"
        },
    }


def prompt(input_sample):
    return {
        "prompt": 'Given the following Arabic tweet, label it as "OFF" if it contains offensive content, or label it as "NOT_OFF" otherwise, based on the content of the tweet. Provide only label.\n\n'
        + "sentence: "
        + input_sample
        + "label: "
    }


def post_process(response):
    label = response["outputs"].strip()
    label = label.replace("<s>", "")
    label = label.replace("</s>", "")

    return label
