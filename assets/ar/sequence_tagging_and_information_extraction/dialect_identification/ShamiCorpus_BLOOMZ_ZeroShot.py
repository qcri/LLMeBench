import os

from llmebench.datasets import ShamiDataset
from llmebench.models import PetalsModel
from llmebench.tasks import DialectIDTask


def config():
    return {
        "dataset": ShamiDataset,
        "dataset_args": {},
        "task": DialectIDTask,
        "task_args": {},
        "model": PetalsModel,
        "model_args": {
            "api_url": os.environ["API_URL"],
            "class_labels": ["Lebanese", "Jordanian", "Palestinian", "Syrian"],
            "max_tries": 22,
        },
        "general_args": {
            "data_path": "data/dialect-data/shami-corpus",
        },
    }


def prompt(input_sample):
    prompt_string = f"Task Description: You are an expert in identifying the dialect of a given arabic text. You will be given a text and you should output the dialect to which the text belongs.\nNote: Please make sure that the class that you output is one of the following: Lebanese, Jordanian, Palestinian, or Syrian.\n Output the class only without any illustrations\nInput:{input_sample} \nLabel: "

    return {
        "prompt": prompt_string,
    }


def post_process(response):
    label = response["outputs"].strip()
    # label = label.replace("<s>", "")
    label = label.replace("</s>", "")
    # label = label.replace("Dialect: ", "").replace("dialect: ", "")
    # label = label.replace("label: ", "")
    # label = label.strip()

    return label
