import random

from llmebench.datasets import SANADAkhbaronaDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import NewsCategorizationTask


random.seed(1333)


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. 3 samples where chosen per test sample based on MaxMarginalRelevance for few shot learning.",
        "scores": {"Accuracy": "0.892"},
    }


def config():
    return {
        "dataset": SANADAkhbaronaDataset,
        "dataset_args": {},
        "task": NewsCategorizationTask,
        "task_args": {},
        "model": OpenAIModel,
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
            "max_tries": 30,
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
