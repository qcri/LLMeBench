import os
import random
import re

from arabic_llm_benchmark.datasets import FactualityCOVID19Dataset
from arabic_llm_benchmark.models import GPTChatCompletionModel
from arabic_llm_benchmark.tasks import FactualityCOVID19Task


random.seed(1333)


def config():
    return {
        "dataset": FactualityCOVID19Dataset,
        "dataset_args": {},
        "task": FactualityCOVID19Task,
        "task_args": {},
        "model": GPTChatCompletionModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
            "class_labels": ["yes", "no"],
            "max_tries": 30,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/factuality_covid19/covid19_infodemic_arabic_data_factuality_binary_test.tsv",
            "fewshot": {
                "train_data_path": "data/factuality_disinformation_harmful_content/factuality_covid19/covid19_infodemic_arabic_data_factuality_binary_train.tsv",
            },
        },
    }


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n"
    out_prompt = out_prompt + "Here are some examples:\n\n"

    for index, example in enumerate(examples):
        out_prompt = (
            out_prompt
            + "Example "
            + str(index)
            + ":"
            + "\n"
            + "tweet: "
            + example["input"]
            + "\nlabel: "
            + example["label"]
            + "\n\n"
        )

    # Append the sentence we want the model to predict for but leave the Label blank
    out_prompt = out_prompt + "tweet: " + input_sample + "\nlabel: \n"

    return out_prompt


def prompt(input_sample, examples):
    base_prompt = f'Annotate the "tweet" into one of the following categories: correct or incorrect. Provide only label.'
    return [
        {
            "role": "system",
            "content": "You are a social media expert, a fact-checker and you can annotate tweets.",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"]

    if label.startswith("I am unable to verify".lower()) or label.startswith(
        "I am unable to categorize".lower()
    ):
        label_fixed = None
    elif "label: incorrect" in label or "incorrect" in label or label == "no":
        label_fixed = "no"
    elif "label: correct" in label or "correct" in label or label == "yes":
        label_fixed = "yes"
    else:
        label_fixed = None

    return label_fixed
