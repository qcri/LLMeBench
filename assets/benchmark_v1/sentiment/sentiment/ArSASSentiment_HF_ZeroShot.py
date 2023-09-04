import os

from llmebench.datasets import ArSASSentimentDataset
from llmebench.models import HuggingFaceInferenceAPIModel
from llmebench.tasks import SentimentTask


def config():
    return {
        "dataset": ArSASSentimentDataset,
        "dataset_args": {},
        "task": SentimentTask,
        "task_args": {},
        "model": HuggingFaceInferenceAPIModel,
        "model_args": {
            "inference_api_url": "https://api-inference.huggingface.co/models/CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment",
            "api_token": os.environ["HUGGINGFACE_API_TOKEN"],
            "max_tries": 5
        },
        "general_args": {
            "data_path": "data/sentiment_emotion_others/sentiment/ArSAS-test.txt"
        },
    }


def prompt(input_sample):
    return [
        {"text": input_sample}
    ]


def post_process(response):
    try:
        scores = [(c['label'], c['score']) for c in response[0]]
        label = sorted(scores, key=lambda x: x[1])[-1][0]
        return upper(label[0]) + lower(label[1:])
    except Exception:
        pass
