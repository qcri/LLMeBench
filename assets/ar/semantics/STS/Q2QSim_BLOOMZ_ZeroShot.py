import re

from llmebench.datasets import STSQ2QDataset
from llmebench.models import PetalsModel
from llmebench.tasks import Q2QSimDetectionTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "bloomz-176b (8bit quantized)",
        "description": "Locally hosted BLOOMZ 176b model (8 bit quantized version) using the Petals.",
        "scores": {"Micro-F1": "0.910"},
    }


def config():
    return {
        "dataset": STSQ2QDataset,
        "task": Q2QSimDetectionTask,
        "model": PetalsModel,
        "model_args": {
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    q1, q2 = input_sample.split("\t")
    # prompt = f"Are the following two questions semantically similar (i.e., asking for similar information)? The output should be exactly in form yes or no.\n\n{input_sample}"
    prompt = f"Are the two questions below semantically similar (i.e., asking for similar information)? The output should be exactly in form yes or no.\n\nQ1: {q1}\nQ2: {q2}\nlabel: "

    return {"prompt": prompt}


def post_process(response):
    input_label = (
        response["outputs"].replace("<s>", "").replace("</s>", "").strip().lower()
    )
    pred_label = ""

    if "yes" in input_label or "label: 1" in input_label:
        pred_label = "1"
    if (
        input_label == "no"
        or input_label.startswith("no,")
        or "label: 0" in input_label
        or "label: no" in input_label
        or "not semantically similar" in input_label
    ):
        pred_label = "0"

    if pred_label == "":
        pred_label = None

    return pred_label
