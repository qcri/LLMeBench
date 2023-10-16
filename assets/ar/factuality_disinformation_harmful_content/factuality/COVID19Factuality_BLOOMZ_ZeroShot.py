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
        f'Does the following tweet contain a factually correct claim or not? Answer only by yes or no.\n\n'
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

    if (label.startswith("i am unable to verify") or label.startswith(
            "i am unable to categorize") or label.startswith("i cannot") or "cannot" in label
    ):
        label_fixed = None
    elif "label: incorrect" in label or "incorrect" in label or label == "no" or label == "لا":
        label_fixed = "no"
    elif "label: correct" in label or "correct" in label or "yes" in label or "نعم" in label:
        label_fixed = "yes"
    else:
        label_fixed = None

    return label_fixed
