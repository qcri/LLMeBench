import os

from llmebench.datasets import ArSASDataset
from llmebench.models import GPTModel
from llmebench.tasks import SentimentTask


def config():
    return {
        "dataset": ArSASDataset,
        "dataset_args": {},
        "task": SentimentTask,
        "task_args": {},
        "model": GPTModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
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
