import os

from llmebench.datasets import ArapTweetDataset
from llmebench.models import LegacyOpenAIModel
from llmebench.tasks import DemographyGenderTask


def config():
    return {
        "dataset": ArapTweetDataset,
        "dataset_args": {},
        "task": DemographyGenderTask,
        "task_args": {},
        "model": LegacyOpenAIModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
            "class_labels": ["Female", "Male"],
            "max_tries": 20,
        },
        "general_args": {
            "data_path": "data/demographic_attributes/gender/test-ARAP-unique.txt"
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
