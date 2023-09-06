import os

from llmebench.datasets import COVID19FactualityDataset
from llmebench.models import BLOOMPetalModel
from llmebench.tasks import FactualityTask


def config():
    return {
        "dataset": COVID19FactualityDataset,
        "dataset_args": {},
        "task": FactualityTask,
        "task_args": {},
        "model": BLOOMPetalModel,
        "model_args": {
            "api_url": os.environ["API_URL"],
            "class_labels": ["yes", "no"],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/factuality_covid19/covid19_infodemic_arabic_data_factuality_binary_test.tsv",
        },
    }


def prompt(input_sample):
    arr = input_sample.split()
    if len(arr) > 1000:
        input_sample = arr[:1000]

    prompt_string = (
        f"Classify following the tweet as yes or no.\n"
        f"Provide only label.\n\n"
        f"tweet: {input_sample}\n"
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

    if label.startswith("I am unable to verify".lower()) or label.startswith(
        "I am unable to categorize".lower()
    ):
        label_fixed = None
    elif "label: incorrect" in label or "incorrect" in label or label == "no":
        label_fixed = "no"
    elif "label: correct" in label or "correct" in label or label == "yes":
        label_fixed = "yes"
    else:
        label_fixed = None

    return label_fixed
