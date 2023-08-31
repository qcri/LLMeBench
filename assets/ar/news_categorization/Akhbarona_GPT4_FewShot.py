import os
import random

from llmebench.datasets import NewsCatAkhbaronaDataset
from llmebench.models import GPTChatCompletionModel
from llmebench.tasks import NewsCatAkhbaronaTask


random.seed(1333)


def config():
    return {
        "dataset": NewsCatAkhbaronaDataset,
        "dataset_args": {},
        "task": NewsCatAkhbaronaTask,
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
            "data_path": "data/news_categorization/SANAD_akhbarona_news_cat_test.tsv",
            "fewshot": {
                "train_data_path": "data/news_categorization/SANAD_akhbarona_news_cat_train.tsv"
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
    if label_fixed.startswith("culture"):
        label_fixed = label_fixed.split("(")[0]

        label_fixed = label_fixed.replace("culture.", "culture")

    return label_fixed
