import os

from arabic_llm_benchmark.datasets import CovidClaimDataset
from arabic_llm_benchmark.models import BLOOMPetalModel
from arabic_llm_benchmark.tasks import ClaimDetectionTask


def config():
    return {
        "dataset": CovidClaimDataset,
        "dataset_args": {},
        "task": ClaimDetectionTask,
        "task_args": {},
        "model": BLOOMPetalModel,
        "model_args": {
            "api_url": os.environ["API_URL"],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/claim_covid19/CT22_arabic_1B_claim_test_gold.tsv"
        },
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
