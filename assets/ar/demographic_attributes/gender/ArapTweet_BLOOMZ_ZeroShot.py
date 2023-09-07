import os

from llmebench.datasets import ArapTweetDataset
from llmebench.models import PetalsModel
from llmebench.tasks import DemographyGenderTask


def config():
    return {
        "dataset": ArapTweetDataset,
        "dataset_args": {},
        "task": DemographyGenderTask,
        "task_args": {},
        "model": PetalsModel,
        "model_args": {
            "api_url": os.environ["API_URL"],
            "class_labels": ["Female", "Male"],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/demographic_attributes/gender/test-ARAP-unique.txt"
        },
    }


def prompt(input_sample):
    prompt_string = (
        f"You are an expert to identify the gender from a person's name.\n\n"
        f"Identify the gender from the following name as 'Female' or 'Male'.\n"
        f"Provide only label.\n\n"
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

    if (
        "female" in label
        or "female." in label
        or "\nfemale" in label
        or label == "female"
    ):
        label = "Female"
    elif "male" in label or "male." in label or "\nmale" in label or label == "male":
        label = "Male"
    else:
        label = None

    return label
