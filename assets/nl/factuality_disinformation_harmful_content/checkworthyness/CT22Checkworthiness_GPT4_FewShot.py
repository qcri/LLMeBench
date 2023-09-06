import os
import re

from llmebench.datasets import CT22CheckworthinessDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import CheckworthinessTask


def config():
    return {
        "dataset": CT22CheckworthinessDataset,
        "dataset_args": {},
        "task": CheckworthinessTask,
        "task_args": {},
        "model": OpenAIModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
            "class_labels": ["0", "1"],
            "max_tries": 30,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/checkworthyness/dutch/CT22_dutch_1A_checkworthy_test_gold.tsv",
            "fewshot": {
                "train_data_path": "data/factuality_disinformation_harmful_content/checkworthyness/dutch/CT22_dutch_1A_checkworthy_train.tsv",
            },
        },
    }


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n"
    out_prompt = out_prompt + "Here are some examples:\n\n"
    for index, example in enumerate(examples):
        label = "no" if example["label"] == "0" else "yes"

        out_prompt = (
            out_prompt
            + "Example "
            + str(example["input_id"])
            + ":"
            + "\n"
            + "tweet: "
            + example["input"]
            + "\nlabel: "
            + label
            + "\n\n"
        )

    # Append the sentence we want the model to predict for but leave the Label blank
    out_prompt = out_prompt + "tweet: " + input_sample + "\nlabel: \n"

    return out_prompt


def prompt(input_sample, examples):
    base_prompt = f'Annotate the "tweet" into "one" of the following categories: checkworthy or not_checkworthy. Provide only label.'
    return [
        {
            "role": "system",
            "content": "As an AI system, your role is to analyze tweets and classify them as 'checkworthy' or 'not_checkworthy' based on their potential importance for journalists and fact-checkers.",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"]

    label = label.replace("label:", "").strip()

    if "label: " in label:
        arr = label.split("label: ")
        label = arr[1].strip()

    if label == "checkworthy" or label == "Checkworthy":
        label_fixed = "1"
    elif label == "Not_checkworthy." or label == "not_checkworthy":
        label_fixed = "0"
    elif "not_checkworthy" in label or "label: not_checkworthy" in label:
        label_fixed = "0"
    elif "checkworthy" in label or "label: checkworthy" in label:
        label_fixed = "1"
    else:
        label_fixed = None

    return label_fixed
