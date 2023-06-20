import os

from arabic_llm_benchmark.datasets import OffensiveDataset
from arabic_llm_benchmark.models import BLOOMPetalModel
from arabic_llm_benchmark.tasks import OffensiveTask


def config():
    return {
        "dataset": OffensiveDataset,
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
            "data_path": "data/sentiment_emotion_others/offensive_language/OSACT2020-sharedTask-test-tweets-labels.txt"
        },
    }


def prompt(input_sample):
    return {
        "prompt": 'Given the following Arabic tweet, label it as "OFF" if it contains offensive content, or label it as "NOT_OFF" otherwise, based on the content of the tweet. Provide only label.\n\n'
        + "sentence: "
        + input_sample
        + "label: \n"
    }


def post_process(response):
    label = response["outputs"].strip()
    label = label.replace("<s>", "")
    label = label.replace("</s>", "")

    return label
