import os
import random

from arabic_llm_benchmark.datasets import NewsCatAkhbaronaDataset
from arabic_llm_benchmark.models import GPTModel
from arabic_llm_benchmark.tasks import NewsCatAkhbaronaTask

random.seed(1333)


def config():
    return {
        "dataset": NewsCatAkhbaronaDataset,
        "dataset_args": {},
        "task": NewsCatAkhbaronaTask,
        "task_args": {},
        "model": GPTModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
            "class_labels": [
                "politics",
                "religion",
                "medical",
                "sports",
                "tech",
                "finance",
                "culture",
            ],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/news_categorization/SANAD_akhbarona_news_cat_test.tsv"
        },
    }


def prompt(input_sample):
    prompt_string = (
        f"Classify the following news article into only one of the following categories: politics, religion, medical, sports, tech, finance, or culture.\n\n"
        f"article: {input_sample}\n"
        f"category: \n"
    )

    print(prompt_string)

    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": prompt_string,
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
