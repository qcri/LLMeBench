import re

from llmebench.datasets import CT22HarmfulDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import HarmfulDetectionTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'.",
        "scores": {"F1 (POS)": "0.533"},
    }


def config():
    return {
        "dataset": CT22HarmfulDataset,
        "task": HarmfulDetectionTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["0", "1"],
            "max_tries": 30,
        },
        "general_args": {"test_split": "ar"},
    }


def prompt(input_sample):
    prompt_string = (
        f'Given the following tweet, analyze its content and identify if it could potentially cause harm to an individual, society, or an organization. Annotate the tweet as "harmful" if it has the potential to cause harm, or "not_harmful" if it does not.\n\n'
        f"tweet: {input_sample}\n"
        f"label: \n"
    )
    return [
        {
            "role": "system",
            "content": "As a system, you are an AI-based social media expert. You are capable of identifying and annotating tweets that could potentially cause harm to individuals, society, or organizations. Your role is to analyze the content of each tweet provided to you, assess its potential impact, and categorize it as harmful or not based on your advanced understanding of harmful speech patterns, social contexts, and potential implications.",
        },
        {
            "role": "user",
            "content": prompt_string,
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"]

    if "label: harmful" in label:
        label_fixed = "1"
    elif "label: not_harmful" in label:
        label_fixed = "0"
    elif "label: " in label:
        arr = label.split("label: ")
        label = arr[1].strip()

    elif label.startswith("harmful") or label.startswith("yes"):
        label_fixed = "1"

    elif (
        label.startswith("no")
        or label == "label: safe"
        or label == "not_harmful"
        or "not harmful" in label
    ):
        label_fixed = "0"
    else:
        label = label.replace(".", "").strip().lower()
        label = label.replace("label:", "").strip()
        label_fixed = label

    return label_fixed
