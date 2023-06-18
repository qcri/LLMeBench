import os

from arabic_llm_benchmark.datasets import ArabGendDataset
from arabic_llm_benchmark.models import BLOOMPetalModel
from arabic_llm_benchmark.tasks import DemographyGenderTask


def config():
    return {
        "dataset": ArabGendDataset,
        "dataset_args": {},
        "task": DemographyGenderTask,
        "task_args": {},
        "model": BLOOMPetalModel,
        "model_args": {
            "api_url": os.environ["API_URL"],
            "class_labels": ["m", "f"],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/demographic_attributes/gender/gender-test.txt"
        },
    }


def prompt(input_sample):
    prompt_string = (
        f"You are an expert to identify the gender from a person's name.\n\n"
        f"Identify the gender from the following name as 'Female' or 'Male'.\n"
        f"name: {input_sample}\n"
        f"label: \n"
    )

    return {
        "prompt": prompt_string,
    }


def post_process(response):
    label = response["outputs"].strip()
    label = label.replace("<s>", "")
    label = label.replace("</s>", "")
    label = label.lower()

    if "Female." in label or "\nFemale" in label:
        label = "f"
    elif "Male." in label or "\nMale" in label:
        label = "m"
    elif label.startswith("I'm sorry, but"):
        label = None
    else:
        label = None

    return label
