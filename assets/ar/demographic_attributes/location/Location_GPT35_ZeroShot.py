import os

from llmebench.datasets import LocationDataset
from llmebench.models import LegacyOpenAIModel
from llmebench.tasks import DemographyLocationTask


def config():
    return {
        "dataset": LocationDataset,
        "dataset_args": {},
        "task": DemographyLocationTask,
        "task_args": {},
        "model": LegacyOpenAIModel,
        "model_args": {
            "class_labels": [
                "ae",
                "OTHERS",
                "bh",
                "dz",
                "eg",
                "iq",
                "jo",
                "kw",
                "lb",
                "ly",
                "ma",
                "om",
                "ps",
                "qa",
                "sa",
                "sd",
                "so",
                "sy",
                "tn",
                "UNK",
                "ye",
                "mr",
            ],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/demographic_attributes/location/arab+others.txt"
        },
    }


def prompt(input_sample):
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": f"Map the following locations to one of the Arab countries. Write ONLY the country code in ISO 3166-1 alpha-2 format without explanation. If the country is outside Arab countries, write ONLY 'OTHERS', and if the location cannot be mapped to any country in the world, write ONLY 'UNK' without any explanation\n {input_sample}",
            }
        ],
    }


def post_process(response):
    out = response["choices"][0]["text"]
    return out.lower()
