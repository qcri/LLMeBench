from llmebench.datasets import UnifiedFCFactualityDataset
from llmebench.models import PetalsModel
from llmebench.tasks import FactualityTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "bloomz-176b (8bit quantized)",
        "description": "Locally hosted BLOOMZ 176b model (8 bit quantized version) using the Petals.",
        "scores": {"Macro-F1": "0.460"},
    }


def config():
    return {
        "dataset": UnifiedFCFactualityDataset,
        "task": FactualityTask,
        "model": PetalsModel,
        "model_args": {
            "class_labels": ["true", "false"],
            "max_tries": 3,
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
