from llmebench.datasets import CT22ClaimDataset
from llmebench.models import PetalsModel
from llmebench.tasks import ClaimDetectionTask


def config():
    return {
        "dataset": CT22ClaimDataset,
        "dataset_args": {},
        "task": ClaimDetectionTask,
        "task_args": {},
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
