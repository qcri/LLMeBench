import random

from llmebench.datasets import SANADAkhbaronaDataset
from llmebench.models import PetalsModel
from llmebench.tasks import NewsCategorizationTask

random.seed(1333)


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "bloomz-176b (8bit quantized)",
        "description": "Locally hosted BLOOMZ 176b model (8 bit quantized version) using the Petals.",
        "scores": {"Accuracy": "0.582"},
    }


def config():
    return {
        "dataset": SANADAkhbaronaDataset,
        "task": NewsCategorizationTask,
        "model": PetalsModel,
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
            "max_tries": 10,
            "max_tokens": 8000,
        },
    }


def prompt(input_sample):
    arr = input_sample.split()
    # if len(arr) > 1000:
    #     article = " ".join(arr[:1000])

    if len(arr) > 800:
        article = " ".join(arr[:800])
    else:
        article = " ".join(arr)
    prompt_string = (
        f"You are an expert news editor and you can categorize news articles.\n\n"
        f'Categorize the following news "article" into one of the following categories: politics, religion, medical, sports, tech, finance, culture\n'
        f"Provide only label and in English.\n\n"
        f"article: {article}\n"
        f"label: \n"
    )
    return {"prompt": prompt_string}


def post_process(response):
    label = response["outputs"].strip().lower()
    label = label.replace("<s>", "")
    label = label.replace("</s>", "")

    label_fixed = label.lower()
    label_fixed = label_fixed.replace("category: ", "")
    label_fixed = label_fixed.replace("science/physics", "tech")
    label_fixed = label_fixed.replace("health/nutrition", "medical")
    label_fixed = label_fixed.replace("nutrition", "medical")
    label_fixed = label_fixed.replace("health", "medical")
    if len(label_fixed.split("\s+")) > 1:
        label_fixed = label_fixed.split("\s+")[0]
    label_fixed = random.choice(label_fixed.split("/")).strip()
    if "science/physics" in label_fixed:
        label_fixed = label_fixed.replace("science/physics", "tech")
    elif "science and technology" in label:
        label_fixed = "tech"
    elif label_fixed.startswith("culture"):
        label_fixed = label_fixed.split("(")[0]

        label_fixed = label_fixed.replace("culture.", "culture")
    else:
        label_fixed = None

    return label_fixed
