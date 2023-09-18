import random

from llmebench.datasets import ASNDDataset
from llmebench.models import PetalsModel
from llmebench.tasks import NewsCategorizationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "bloomz-176b (8bit quantized)",
        "description": "Locally hosted BLOOMZ 176b model (8 bit quantized version) using the Petals.",
        "scores": {"Macro-F1": "0.134"},
    }


def config():
    return {
        "dataset": ASNDDataset,
        "dataset_args": {},
        "task": NewsCategorizationTask,
        "task_args": {},
        "model": PetalsModel,
        "model_args": {
            "class_labels": [
                "crime-war-conflict",
                "spiritual",
                "health",
                "politics",
                "human-rights-press-freedom",
                "education",
                "business-and-economy",
                "art-and-entertainment",
                "others",
                "science-and-technology",
                "sports",
                "environment",
            ],
            "max_tries": 10,
        },
    }


def prompt(input_sample):
    arr = input_sample.split()

    if len(arr) > 1000:
        article = " ".join(arr[:1000])
    else:
        article = " ".join(arr)

    prompt_string = (
        f"You are an expert news editor and know how to categorize news articles.\n\n"
        f"Categorize the following tweet into one of the following categories: "
        f"crime-war-conflict, spiritual, health, politics, human-rights-press-freedom, "
        f"education, business-and-economy, art-and-entertainment, others, "
        f"science-and-technology, sports, environment\n"
        f"Provide only label and in English.\n\n"
        f"\ntweet: {article}"
        f"\ncategory: \n"
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
