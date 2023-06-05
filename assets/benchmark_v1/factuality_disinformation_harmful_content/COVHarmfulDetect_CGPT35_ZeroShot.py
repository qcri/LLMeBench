import os

from arabic_llm_benchmark.datasets import CovidHarmfulDataset
from arabic_llm_benchmark.models import GPTModel
from arabic_llm_benchmark.tasks import HarmfulDetectionTask


def config():
    return {
        "dataset": CovidHarmfulDataset,
        "dataset_args": {},
        "task": HarmfulDetectionTask,
        "task_args": {},
        "model": GPTModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": "gpt",
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "arabic_llm_benchmark/data/factuality_disinformation_harmful_content/harmful"
                         "/CT22_arabic_1C_harmful_test_gold.tsv"},
    }


def prompt(input_sample):
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": "Classify the sentence as harmful or not harmful. Answer only by yes or no. Provide only "
                        "label.\n\n" +
                        "sentence: " + input_sample +
                        "label: \n"
            }
        ],
    }


def post_process(response):
    pred_label = response["choices"][0]["text"]
    pred_label = pred_label.replace(".", "").strip().lower()

    if pred_label.startswith("harmful") or pred_label.startswith("yes"):
        pred_label = "1"

    if pred_label.startswith("no") or pred_label == "label: safe" or "not harmful" in pred_label:
        pred_label = "0"

    return pred_label
