import random

from llmebench.datasets import SANADAlKhaleejDataset
from llmebench.models import FastChatModel
from llmebench.tasks import NewsCategorizationTask


random.seed(1333)


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Jais-13b-chat",
        "description": "Locally hosted Jais-13b-chat model using FastChat.",
    }


def config():
    return {
        "dataset": SANADAlKhaleejDataset,
        "task": NewsCategorizationTask,
        "model": FastChatModel,
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
    base_prompt = (
        f'Categorize the news "article" into one of the following categories: culture, finance, medical, politics, religion, sports, tech\n\n'
        f"article: {input_sample}\n"
        f"category: \n"
    )
    return [
        {
            "role": "user",
            "content": base_prompt,
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"]
    label_list = config()["model_args"]["class_labels"]
    label_fixed = label.lower()
    label_fixed = label_fixed.replace("category: ", "")
    label_fixed = label_fixed.replace("science/physics", "tech")
    label_fixed = label_fixed.replace("health/nutrition", "medical")

    if "سياسة" in label or "السياسة" in label:
        label_fixed = "politics"

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
