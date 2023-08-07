import os

from arabic_llm_benchmark.datasets import SpamDataset
from arabic_llm_benchmark.models import BLOOMPetalModel
from arabic_llm_benchmark.tasks import SpamTask


def config():
    return {
        "dataset": SpamDataset,
        "dataset_args": {},
        "task": SpamTask,
        "task_args": {},
        "model": BLOOMPetalModel,
        "model_args": {
            "api_url": os.environ["API_URL"],
            "class_labels": ["__label__ADS", "__label__NOTADS"],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/sentiment_emotion_others/spam/ArabicAds-test.txt"
        },
    }


def prompt(input_sample):
    return {
        "prompt": "Classify the following sentence as 'spam' or 'not_spam'. Provide only label.\n\n"
        + "sentence: "
        + input_sample
        + "label: "
    }


def post_process(response):
    label = response["outputs"].strip()
    label = label.replace("<s>", "")
    label = label.replace("</s>", "")
    if label == "spam":
        label = "__label__ADS"
    else:
        label = "__label__NOTADS"

    return label
