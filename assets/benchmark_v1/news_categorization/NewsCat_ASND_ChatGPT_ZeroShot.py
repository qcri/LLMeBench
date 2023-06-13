import os
import random

from arabic_llm_benchmark.datasets import NewsCatASNDDataset
from arabic_llm_benchmark.models import GPTModel, RandomGPTModel
from arabic_llm_benchmark.tasks import NewsCatASNDTask

random.seed(1333)


def config():
    return {
        "dataset": NewsCatASNDDataset,
        "dataset_args": {},
        "task": NewsCatASNDTask,
        "task_args": {"test": "useless"},
        "model": GPTModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": "gpt",
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
                "sports or environment",
            ],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/news_categorization/Arabic_Social_Media_News_Dataset_ASND/sm_news_ar_tst.csv"
        },
    }


def prompt(input_sample):
    prompt_string = (
        f"Classify the following tweet into one of the following categories: "
        f"crime-war-conflict, spiritual, health, politics, human-rights-press-freedom, "
        f"education, business-and-economy, art-and-entertainment, others, "
        f"science-and-technology, sports or environment\n"
        f"\ntweet: {input_sample}"
        f"\ncategory: \n"
    )

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
    label_fixed = label_fixed.replace("text:", "")
    if label_fixed != "true" or label_fixed != "false":
        if len(label_fixed.split()) > 1:
            label_fixed = label_fixed.split()[0]

    return label_fixed
