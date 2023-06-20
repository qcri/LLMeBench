import os
import random
import re

from arabic_llm_benchmark.datasets import FactualityUnifiedFCDataset
from arabic_llm_benchmark.models import BLOOMPetalModel
from arabic_llm_benchmark.tasks import FactualityUnifiedFCTask


random.seed(1333)


def config():
    return {
        "dataset": FactualityUnifiedFCDataset,
        "dataset_args": {},
        "task": FactualityUnifiedFCTask,
        "task_args": {},
        "model": BLOOMPetalModel,
        "model_args": {
            "api_url": os.environ["API_URL"],
            "class_labels": ["true", "false"],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/factuality_stance_ramy/ramy_arabic_fact_checking.tsv"
        },
    }


def prompt(input_sample):
    return {
        "prompt": (
            f'Annotate the "text" into one of the following categories: correct or incorrect\n\n'
            f"tweet: {input_sample}\n"
            f"label: \n"
        )
    }


def post_process(response):
    label = response["outputs"].strip().lower()
    label = label.replace("<s>", "")
    label = label.replace("</s>", "")
    # label_fixed = label.replace("label:", "").strip()

    if (
        label.startswith("I am unable to verify".lower())
        or label.startswith("I am unable to categorize".lower())
        or label.startswith(
            "I am an AI language model and I am unable to verify".lower()
        )
    ):
        label_fixed = None
    elif "label: incorrect" in label or "incorrect" in label:
        label_fixed = "False"
    elif "label: correct" in label or "correct" in label:
        label_fixed = "True"
    else:
        label_fixed = None

    return label_fixed