from llmebench.datasets import ArabGendDataset
from llmebench.models import LegacyOpenAIModel
from llmebench.tasks import ClassificationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-35-turbo (version 0301)",
        "description": "GPT35 model hosted on Azure, using the Completion API. API version '2023-03-15-preview'.",
    }


def config():
    return {
        "dataset": ArabGendDataset,
        "task": ClassificationTask,
        "model": LegacyOpenAIModel,
        "model_args": {
            "class_labels": ["m", "f"],
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": f"If the following person name can be considered as male, write 'm' without explanation, and if it can be considered as female, write 'f' without explanation.\n {input_sample}",
            }
        ],
    }


def post_process(response):
    return response["choices"][0]["text"]
