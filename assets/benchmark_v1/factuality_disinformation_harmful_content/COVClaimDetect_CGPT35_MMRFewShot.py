import os

import pandas as pd

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
            "engine_name": "gpt",
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/claim_covid19/CT22_arabic_1B_claim_test_gold.tsv",
            "fewshot": {
                "train_data_path": "data/factuality_disinformation_harmful_content/claim_covid19/CT22_arabic_1B_claim_train.tsv",
                "n_shots": 3,
            },
        },
    }


def prompt(input_sample, examples):
    base_prompt = "Does this sentence contain a factual claim? Answer only by yes or no. Provide only label.\n"

    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": few_shot_prompt(input_sample, base_prompt, examples),
            }
        ],
    }


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n"
    for example in examples:
        # Found chatgpt confused when using 0 and 1 in the prompt
        label = "no" if example["label"] == "0" else "yes"
        out_prompt = (
            out_prompt + "Sentence: " + example["input"] + "\nLabel: " + label + "\n\n"
        )

    # Append the sentence we want the model to predict for but leave the Label blank
    out_prompt = out_prompt + "Sentence: " + input_sample + "\nLabel: \n"

    # print("=========== FS Prompt =============\n")
    # print(out_prompt)

    return out_prompt


def post_process(response):
    input_label = response["choices"][0]["text"]
    input_label = input_label.replace(".", "").strip().lower()
    pred_label = ""

    if (
        "yes" in input_label
        or "contains a factual claim" in input_label
        or "label: 1" in input_label
    ):
        pred_label = "1"
    if (
        input_label == "no"
        or "label: 0" in input_label
        or "label: no" in input_label
        or "not contain a factual claim" in input_label
        or "doesn't contain a factual claim" in input_label
    ):
        pred_label = "0"

    if pred_label == "":
        pred_label = "0"

    return pred_label
