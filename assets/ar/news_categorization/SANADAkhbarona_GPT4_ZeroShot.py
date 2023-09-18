import random

from llmebench.datasets import SANADAkhbaronaDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import NewsCategorizationTask


random.seed(1333)


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'.",
        "scores": {"Accuracy": "0.877"},
    }


def config():
    return {
        "dataset": SANADAkhbaronaDataset,
        "task": NewsCategorizationTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": [
                "politics",
                "religion",
                "medical",
                "sports",
                "tech",
                "finance",
                "culture",
            ],
            "max_tries": 30,
        },
    }


def prompt(input_sample):
    prompt_string = (
        f'Categorize the news "article" into one of the following categories: politics, religion, medical, sports, tech, finance, culture\n\n'
        f"article: {input_sample}\n"
        f"category: \n"
    )
    return [
        {
            "role": "system",
            "content": "You are an expert news editor and know how to categorize news articles.",
        },
        {
            "role": "user",
            "content": prompt_string,
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"]

    label_fixed = label.lower()
    label_fixed = label_fixed.replace("category: ", "")
    label_fixed = label_fixed.replace("science/physics", "tech")
    label_fixed = label_fixed.replace("health/nutrition", "medical")
    if len(label_fixed.split("\s+")) > 1:
        label_fixed = label_fixed.split("\s+")[0]
    label_fixed = random.choice(label_fixed.split("/")).strip()
    if "science/physics" in label_fixed:
        label_fixed = label_fixed.replace("science/physics", "tech")
    if label_fixed.startswith("culture"):
        label_fixed = label_fixed.split("(")[0]

        label_fixed = label_fixed.replace("culture.", "culture")

    return label_fixed
