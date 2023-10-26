from llmebench.datasets import ArapTweetDataset
from llmebench.models import LegacyOpenAIModel
from llmebench.tasks import ClassificationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-35-turbo (version 0301)",
        "description": "GPT35 model hosted on Azure, using the Completion API. API version '2023-03-15-preview'.",
        "scores": {"Macro-F1": "0.883"},
    }


def config():
    return {
        "dataset": ArapTweetDataset,
        "task": ClassificationTask,
        "model": LegacyOpenAIModel,
        "model_args": {
            "class_labels": ["Female", "Male"],
            "max_tries": 20,
        },
    }


def prompt(input_sample):
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": f"If the following person name can be considered as male, write 'Male' without explanation, and if it can be considered as female, write 'Female' without explanation.\n {input_sample}",
            }
        ],
    }


def post_process(response):
    label = response["choices"][0]["text"]

    if "Female." in label or "\nFemale" in label:
        label = "Female"
    elif "Male." in label or "\nMale" in label:
        label = "Male"
    elif label.startswith("I'm sorry, but"):
        label = None
    else:
        label = None

    return label
