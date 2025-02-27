import os

from llmebench.datasets import ShamiDataset
from llmebench.models import LegacyOpenAIModel
from llmebench.tasks import DialectIDTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-3.5",
        "description": "GPT 3.5 tokens model hosted on Azure, using the ChatCompletion API.",
        "scores": {"Macro-F1": ""},
    }


def config():
    return {
        "dataset": ShamiDataset,
        "task": DialectIDTask,
        "model": LegacyOpenAIModel,
        "model_args": {
            "class_labels": ["Lebanese", "Jordanian", "Palestinian", "Syrian"],
            "max_tries": 3,
        },
        "general_args": {"data_path": "data/dialect-data/shami-corpus"},
    }


def prompt(input_sample):
    prompt_string = f"Task Description: You are an expert in identifying the dialect of a given arabic text. You will be given a text and you should output the dialect to which the text belongs.\nNote: Please make sure that the class that you output is one of the following: Lebanese, Jordanian, Palestinian, or Syrian.\n Output the class only without any illustrations\nInput:{input_sample} \nLabel: "

    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [{"sender": "user", "text": prompt_string}],
    }


def post_process(response):
    label = response["choices"][0]["text"]
    return label
