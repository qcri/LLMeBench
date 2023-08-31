import os
import random

from llmebench.datasets import NewsCatAlArabiyaDataset
from llmebench.models import BLOOMPetalModel
from llmebench.tasks import NewsCatAlArabiyaTask

random.seed(1333)


def config():
    return {
        "dataset": NewsCatAlArabiyaDataset,
        "dataset_args": {},
        "task": NewsCatAlArabiyaTask,
        "task_args": {},
        "model": BLOOMPetalModel,
        "model_args": {
            "api_url": os.environ["API_URL"],
            "class_labels": [
                "culture",
                "finance",
                "medical",
                "politics",
                "religion",
                "sports",
                "tech",
            ],
            "max_tries": 10,
        },
        "general_args": {
            "data_path": "data/news_categorization/SANAD_alarabiya_news_cat_test.tsv"
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
        f'Categorize the news "article" into one of the following categories: culture, finance, medical, politics, religion, sports, tech\n'
        f"Provide only label and in English.\n\n"
        f"article: {article}\n"
        f"category: \n"
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

    return label_fixed
