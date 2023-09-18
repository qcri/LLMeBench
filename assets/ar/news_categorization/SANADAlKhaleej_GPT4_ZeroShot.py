import random

from llmebench.datasets import SANADAlKhaleejDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import NewsCategorizationTask


random.seed(1333)


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'.",
        "scores": {"Accuracy": "0.911"},
    }


def config():
    return {
        "dataset": SANADAlKhaleejDataset,
        "task": NewsCategorizationTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": [
                "culture",
                "finance",
                "medical",
                "politics",
                "religion",
                "sports",
                "tech",
            ],
            "max_tries": 30,
        },
    }


def prompt(input_sample):
    prompt_string = (
        f'Categorize the news "article" into one of the following categories: culture, finance, medical, politics, religion, sports, tech\n\n'
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
    label_list = config()["model_args"]["class_labels"]
    label_fixed = label.lower()
    label_fixed = label_fixed.replace("category: ", "")
    label_fixed = label_fixed.replace("science/physics", "tech")
    label_fixed = label_fixed.replace("health/nutrition", "medical")

    if label_fixed.strip() in label_list:
        label_fixed = label_fixed.strip()

    elif "science/physics" in label_fixed:
        label_fixed = label_fixed.replace("science/physics", "tech")
    elif label_fixed.startswith("culture"):
        label_fixed = label_fixed.split("(")[0]
        label_fixed = label_fixed.replace("culture.", "culture")
    elif "/" in label:
        label_fixed = random.choice(label_fixed.split("/")).strip()
    else:
        label_fixed = None

    return label_fixed
