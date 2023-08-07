import os

from arabic_llm_benchmark.datasets import ArSASSentimentDataset
from arabic_llm_benchmark.models import BLOOMPetalModel
from arabic_llm_benchmark.tasks import SentimentTask


def config():
    return {
        "dataset": ArSASSentimentDataset,
        "dataset_args": {},
        "task": SentimentTask,
        "task_args": {},
        "model": BLOOMPetalModel,
        "model_args": {
            "api_url": os.environ["API_URL"],
            "class_labels": ["Positive", "Negative", "Neutral", "Mixed"],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/sentiment_emotion_others/sentiment/ArSAS-test.txt"
        },
    }


def prompt(input_sample):
    return {
        "prompt": "Choose only one sentiment between: Positive, Negative, Neutral, or Mixed for this Arabic sentence. Provide only label.\n\n"
        + "sentence: "
        + input_sample
        + "label: "
    }


def post_process(response):
    label = response["outputs"].strip()
    label = label.replace("<s>", "")
    label = label.replace("</s>", "")
    return label
