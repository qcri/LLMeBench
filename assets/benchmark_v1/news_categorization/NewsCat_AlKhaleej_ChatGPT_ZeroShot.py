import os
import random

from arabic_llm_benchmark.datasets import NewsCatAlKhaleejDataset
from arabic_llm_benchmark.models import GPTModel
from arabic_llm_benchmark.tasks import NewsCatAlKhaleejTask

random.seed(1333)


def config():
    return {
        "dataset": NewsCatAlKhaleejDataset,
        "dataset_args": {},
        "task": NewsCatAlKhaleejTask,
        "task_args": {},
        "model": GPTModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": "gpt",
            "class_labels": ["sports", "medical", "finance", "tech", "politics", "medical", "sports", "politics", "culture"],  
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/news_categorization/SANAD_alkhaleej_news_cat_test.tsv"
        },
    }


def prompt(input_sample):

    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": f"Classify the following news article into one of the following categories: sports, medical, finance, tech, politics, medical, sports, politics, or culture.\n\narticle: {input_sample}\ncategory: \n",
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
