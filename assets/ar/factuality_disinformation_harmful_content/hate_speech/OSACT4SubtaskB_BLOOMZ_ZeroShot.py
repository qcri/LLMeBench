import os

from llmebench.datasets import OSACT4SubtaskBDataset
from llmebench.models import PetalsModel
from llmebench.tasks import HateSpeechTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "bloomz-176b (8bit quantized)",
        "description": "Locally hosted BLOOMZ 176b model (8 bit quantized version) using the Petals.",
        "scores": {"Macro-F1": "0.503"},
    }


def config():
    return {
        "dataset": OSACT4SubtaskBDataset,
        "dataset_args": {},
        "task": HateSpeechTask,
        "task_args": {},
        "model": PetalsModel,
        "model_args": {
            "class_labels": ["HS", "NOT_HS"],
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    return {
        "prompt": 'Given the following Arabic tweet, label it as "HS" or "NOT_HS" based on the content of the tweet. Provide only label.\n\n'
        + "sentence: "
        + input_sample
        + "label: \n"
    }


def post_process(response):
    label = response["outputs"].strip()
    label = label.replace("<s>", "")
    label = label.replace("</s>", "")

    return label
