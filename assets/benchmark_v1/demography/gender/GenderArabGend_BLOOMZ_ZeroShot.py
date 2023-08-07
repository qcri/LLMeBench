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
        f"Classify the name as male or female. Provide only label. \n\n"
        f"name: {input_sample}\n"
        f"label: "
    )

    return {
        "prompt": prompt_string,
    }


def post_process(response):
    content = response["outputs"].strip()
    content = content.replace("<s>", "")
    content = content.replace("</s>", "")
    label = content.lower()

    if "female" in label:
        return "f"
    elif "male" in label:
        return "m"
    else:
        return None
