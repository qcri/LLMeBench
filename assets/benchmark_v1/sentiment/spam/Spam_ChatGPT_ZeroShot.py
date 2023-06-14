import os

from arabic_llm_benchmark.datasets import SpamDataset
from arabic_llm_benchmark.models import GPTModel, RandomGPTModel
from arabic_llm_benchmark.tasks import SpamTask


def config():
    return {
        "dataset": SpamDataset,
        "dataset_args": {},
        "task": SpamTask,
        "task_args": {},
        "model": GPTModel,
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
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": f"If the following sentence can be classified as spam or contains an advertisemnt, write '__label__ADS' without explnanation, otherwise write '__label__NOTADS' without explanantion.\n {input_sample}",
            }
        ],
    }


def post_process(response):
    out = response["choices"][0]["text"]
    j = out.find(".")
    if j > 0:
        out = out[0:j]
    return out
