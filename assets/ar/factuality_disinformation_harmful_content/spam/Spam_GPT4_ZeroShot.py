import os

from llmebench.datasets import SpamDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import SpamTask


def config():
    return {
        "dataset": SpamDataset,
        "dataset_args": {},
        "task": SpamTask,
        "task_args": {},
        "model": OpenAIModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
            "class_labels": ["__label__ADS", "__label__NOTADS"],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/sentiment_emotion_others/spam/ArabicAds-test.txt"
        },
    }


def prompt(input_sample):
    return [
        {
            "role": "system",
            "content": "You are an AI assistant that helps people find information.",
        },
        {
            "role": "user",
            "content": f"If the following sentence can be classified as spam or contains an advertisemnt, write '__label__ADS' without explnanation, otherwise write '__label__NOTADS' without explanantion.\n {input_sample}\n",
        },
    ]


def post_process(response):
    out = response["choices"][0]["message"]["content"]
    j = out.find(".")
    if j > 0:
        out = out[0:j]
    return out
