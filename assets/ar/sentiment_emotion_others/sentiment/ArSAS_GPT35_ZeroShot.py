import os

from llmebench.datasets import ArSASDataset
from llmebench.models import LegacyOpenAIModel
from llmebench.tasks import SentimentTask


def config():
    return {
        "dataset": ArSASDataset,
        "dataset_args": {},
        "task": SentimentTask,
        "task_args": {},
        "model": LegacyOpenAIModel,
        "model_args": {
            "class_labels": ["Positive", "Negative", "Neutral", "Mixed"],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/sentiment_emotion_others/sentiment/ArSAS-test.txt"
        },
    }


def prompt(input_sample):
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": f"Choose only one sentiment between: Positive, Negative, Neutral, or Mixed for this Arabic sentence: \n {input_sample}",
            }
        ],
    }


def post_process(response):
    out = response["choices"][0]["text"]
    j = out.find(".")
    if j > 0:
        out = out[0:j]
    return out
