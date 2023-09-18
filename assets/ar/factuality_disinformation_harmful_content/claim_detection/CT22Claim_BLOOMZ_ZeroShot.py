from llmebench.datasets import CT22ClaimDataset
from llmebench.models import PetalsModel
from llmebench.tasks import ClaimDetectionTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "bloomz-176b (8bit quantized)",
        "description": "Locally hosted BLOOMZ 176b model (8 bit quantized version) using the Petals.",
        "scores": {"Accuracy": "0.532"},
    }


def config():
    return {
        "dataset": CT22ClaimDataset,
        "task": ClaimDetectionTask,
        "model": PetalsModel,
        "model_args": {
            "max_tries": 3,
        },
        "general_args": {"test_split": "ar"},
    }


def prompt(input_sample):
    return {
        "prompt": "Does this sentence contain a factual claim? Answer only by yes or no. Provide only label.\n\n"
        + "Sentence: "
        + input_sample
        + "\nLabel: \n"
    }


def post_process(response):
    label = response["outputs"].strip().lower()
    label = label.replace("<s>", "")
    label = label.replace("</s>", "")

    pred_label = None

    if "yes" in label or "contains a factual claim" in label or "label: 1" in label:
        pred_label = "1"
    if (
        label == "no"
        or "label: 0" in label
        or "label: no" in label
        or "not contain a factual claim" in label
        or "doesn't contain a factual claim" in label
    ):
        pred_label = "0"

    return pred_label
