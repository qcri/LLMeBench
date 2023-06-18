import os

from arabic_llm_benchmark.datasets import CovidClaimDataset
from arabic_llm_benchmark.models import GPTModel
from arabic_llm_benchmark.tasks import ClaimDetectionTask


def config():
    return {
        "dataset": CovidClaimDataset,
        "dataset_args": {},
        "task": ClaimDetectionTask,
        "task_args": {},
        "model": GPTModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
            "class_labels": ["0", "1"],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/claim_covid19/CT22_arabic_1B_claim_test_gold.tsv"
        },
    }


def prompt(input_sample):
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": "Does this sentence contain a factual claim? Answer only by yes or no."
                "\n\nsentence: " + input_sample + "label: \n",
            }
        ],
    }


def post_process(response):
    pred_label = response["choices"][0]["text"]
    pred_label = pred_label.replace(".", "").strip().lower()

    if pred_label == "yes" or pred_label == "the sentence contains a factual claim":
        pred_label = "1"
    if pred_label == "no":
        pred_label = "0"

    return pred_label
