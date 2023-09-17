import os
import random

from llmebench.datasets import NewsCatAlArabiyaDataset
from llmebench.models import GPTChatCompletionModel
from llmebench.tasks import NewsCatAlArabiyaTask


random.seed(1333)


def config():
    return {
        "dataset": NewsCatAlArabiyaDataset,
        "dataset_args": {},
        "task": NewsCatAlArabiyaTask,
        "task_args": {},
        "model": GPTChatCompletionModel,
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
            "max_tries": 30,
        },
        "general_args": {
            "data_path": "data/news_categorization/SANAD_alarabiya_news_cat_test.tsv",
            "fewshot": {
                "train_data_path": "data/news_categorization/SANAD_alarabiya_news_cat_train.tsv"
            },
        },
    }


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n"
    out_prompt = out_prompt + "Here are some examples:\n\n"

    for index, example in enumerate(examples):
        out_prompt = (
            out_prompt
            + "Example "
            + str(index)
            + ":"
            + "\n"
            + "article: "
            + example["input"]
            + "\ncategory: "
            + example["label"]
            + "\n\n"
        )
    out_prompt = out_prompt + "article: " + input_sample + "\ncategory: \n"

    return out_prompt


def prompt(input_sample, examples):
    base_prompt = f'Categorize the news "article" into one of the following categories: politics, religion, medical, sports, tech, finance, culture'
    return [
        {
            "role": "system",
            "content": "You are an expert news editor and know how to categorize news articles.",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"]

    label_list = config()["model_args"]["class_labels"]
    label_fixed = label.lower()
    label_fixed = label_fixed.replace("category: ", "")
    label_fixed = label_fixed.replace("science/physics", "tech")
    label_fixed = label_fixed.replace("health/nutrition", "medical")

    if label_fixed in label_list:
        label_fixed = label_fixed

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