import random

from llmebench.datasets import SANADAlKhaleejDataset
from llmebench.models import LegacyOpenAIModel
from llmebench.tasks import NewsCategorizationTask

random.seed(1333)


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-35-turbo (version 0301)",
        "description": "GPT35 model hosted on Azure, using the Completion API. API version '2023-03-15-preview'.",
        "scores": {"Accuracy": "0.864"},
    }


def config():
    return {
        "dataset": SANADAlKhaleejDataset,
        "task": NewsCategorizationTask,
        "model": LegacyOpenAIModel,
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
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": f"Classify the following news article into one of the following categories: culture, finance, medical, politics, religion, sports, tech\n\narticle: {input_sample}\ncategory: \n",
            }
        ],
    }


def post_process(response):
    label = response["choices"][0]["text"]
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
