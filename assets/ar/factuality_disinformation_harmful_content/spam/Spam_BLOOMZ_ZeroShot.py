from llmebench.datasets import SpamDataset
from llmebench.models import PetalsModel
from llmebench.tasks import SpamTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "bloomz-176b (8bit quantized)",
        "description": "Locally hosted BLOOMZ 176b model (8 bit quantized version) using the Petals.",
        "scores": {"Macro-F1": "0.152"},
    }


def config():
    return {
        "dataset": SpamDataset,
        "task": SpamTask,
        "model": PetalsModel,
        "model_args": {
            "class_labels": ["__label__ADS", "__label__NOTADS"],
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    return {
        "prompt": "Classify the following sentence as 'spam' or 'not_spam'. Provide only label.\n\n"
        + "sentence: "
        + input_sample
        + "label: "
    }


def post_process(response):
    label = response["outputs"].strip()
    label = label.replace("<s>", "")
    label = label.replace("</s>", "")
    if label == "spam":
        label = "__label__ADS"
    else:
        label = "__label__NOTADS"

    return label
