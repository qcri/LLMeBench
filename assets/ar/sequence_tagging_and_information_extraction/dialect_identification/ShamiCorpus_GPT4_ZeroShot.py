import os

from llmebench.datasets import ShamiDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import DialectIDTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'.",
        "scores": {"Macro-F1": ""},
    }


def config():
    return {
        "dataset": ShamiDataset,
        "task": DialectIDTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["Lebanese", "Jordanian", "Palestinian", "Syrian"],
            "max_tries": 3,
        },
        "general_args": {"data_path": "data/dialect-data/shami-corpus"},
    }


def prompt(input_sample):
    prompt_string = f"Task Description: You are an expert in identifying the dialect of a given arabic text. You will be given a text and you should output the dialect to which the text belongs.\nNote: Please make sure that the class that you output is one of the following: Lebanese, Jordanian, Palestinian, or Syrian.\n Output the class only without any illustrations\nInput:{input_sample} \nLabel: "

    return [
        {
            "role": "system",
            "content": "You are an AI assistant that helps people find information.",
        },
        {
            "role": "user",
            "content": prompt_string,
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"]
    return label
