import os
import random

from arabic_llm_benchmark.datasets import NewsCatASNDDataset
from arabic_llm_benchmark.models import GPTChatCompletionModel
from arabic_llm_benchmark.tasks import NewsCatASNDTask


random.seed(1333)


def config():
    return {
        "dataset": NewsCatASNDDataset,
        "dataset_args": {},
        "task": NewsCatASNDTask,
        "task_args": {},
        "model": GPTChatCompletionModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": "qvoice-gpt4",
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
            "max_tries": 30,
        },
        "general_args": {
            "data_path": "data/news_categorization/Arabic_Social_Media_News_Dataset_ASND/sm_news_ar_tst.csv"
        },
    }


def prompt(input_sample):
    prompt_string = (
        f"Categorize the following tweet into one of the following categories: "
        f"crime-war-conflict, spiritual, health, politics, human-rights-press-freedom, "
        f"education, business-and-economy, art-and-entertainment, others, "
        f"science-and-technology, sports, environment\n"
        f"\ntweet: {input_sample}"
        f"\ncategory: \n"
    )

    return [
        {
            "role": "system",
            "content": "You are an expert tweet annotator and know how to categorize news tweet.",
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
    label_fixed = label_fixed.replace("text:", "")
    if label_fixed != "true" or label_fixed != "false":
        if len(label_fixed.split()) > 1:
            label_fixed = label_fixed.split()[0]

    return label_fixed
