import os

from llmebench.datasets import BanglaNewsCategorizationDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import ClassificationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-3.5-turbo",
        "description": "gpt-3.5-turbo model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. 3 samples where chosen per test sample based on MaxMarginalRelevance for few shot learning.",
    }


def config():
    return {
        "dataset": BanglaNewsCategorizationDataset,
        "task": ClassificationTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": [
                "entertainment",
                "state",
                "sports",
                "national",
                "kolkata",
                "international",
            ],
            "max_tries": 20,
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
            + "news: "
            + example["input"]
            + "\nlabel: "
            + example["label"]
            + "\n\n"
        )

    out_prompt = out_prompt + "news: " + input_sample + "\nlabel: \n"

    return out_prompt


def prompt(input_sample, examples):
    base_prompt = 'Annotate the "news" into "one" of the following categories: "entertainment", "state", "sports", "national", "kolkata", or "international"'
    return [
        {
            "role": "system",
            "content": f"You are a expert annotator. Your task is to analyze the news and identify the appropriate category of the news.\n",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"]

    label_fixed = label.replace("label:", "").strip()
    if label_fixed.startswith("Please provide the text"):
        label_fixed = None

    return label_fixed
