import re

from llmebench.datasets import CT22CheckworthinessDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import CheckworthinessTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. 3 samples where chosen per test sample based on MaxMarginalRelevance for few shot learning.",
    }


def config():
    return {
        "dataset": CT22CheckworthinessDataset,
        "task": CheckworthinessTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["0", "1"],
            "max_tries": 30,
        },
        "general_args": {"test_split": "es", "fewshot": {"train_split": "es"}},
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
