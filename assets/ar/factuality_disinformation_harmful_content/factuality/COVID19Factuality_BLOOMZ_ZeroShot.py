from llmebench.datasets import COVID19FactualityDataset
from llmebench.models import PetalsModel
from llmebench.tasks import FactualityTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "bloomz-176b (8bit quantized)",
        "description": "Locally hosted BLOOMZ 176b model (8 bit quantized version) using the Petals.",
    }


def config():
    return {
        "dataset": COVID19FactualityDataset,
        "task": FactualityTask,
        "model": PetalsModel,
        "model_args": {
            "class_labels": ["yes", "no"],
            "max_tries": 3,
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
