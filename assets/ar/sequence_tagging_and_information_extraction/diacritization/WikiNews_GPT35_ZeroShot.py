from llmebench.datasets import WikiNewsDiacritizationDataset
from llmebench.models import LegacyOpenAIModel
from llmebench.tasks import ArabicDiacritizationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-35-turbo (version 0301)",
        "description": "GPT35 model hosted on Azure, using the Completion API. API version '2023-03-15-preview'.",
        "scores": {"WER": "0.308"},
    }


def config():
    return {
        "dataset": WikiNewsDiacritizationDataset,
        "dataset_args": {},
        "task": ArabicDiacritizationTask,
        "task_args": {},
        "model": LegacyOpenAIModel,
        "model_args": {
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": f"Diacritize fully the following Arabic sentence: {input_sample}",
            }
        ],
    }


def post_process(response):
    return response["choices"][0]["text"]
