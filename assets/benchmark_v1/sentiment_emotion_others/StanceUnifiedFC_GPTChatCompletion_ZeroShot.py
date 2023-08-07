import os
import random
import re

from arabic_llm_benchmark.datasets import StanceUnifiedFCDataset
from arabic_llm_benchmark.models import GPTChatCompletionModel
from arabic_llm_benchmark.tasks import StanceUnifiedFCTask


random.seed(1333)


def config():
    return {
        "dataset": StanceUnifiedFCDataset,
        "dataset_args": {},
        "task": StanceUnifiedFCTask,
        "task_args": {},
        "model": GPTChatCompletionModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
            "class_labels": ["agree", "disagree", "unrelated"],
            "max_tries": 30,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/factuality_stance_ramy/ramy_arabic_stance.jsonl"
        },
    }


def prompt(input_sample):
    article = input_sample["article"]
    # article_arr = article.split()
    # if len(article_arr) > 2200:
    #     article_str = " ".join(article_arr[:2200])
    # else:
    article_str = article

    prompt_string = (
        f"Based on your analysis, determine the stance of the news article towards the claim. The possible stances could be 'agree', 'disagree', or 'unrelated'."
        f"\n\n"
        f"claim: {input_sample['claim']}\n"
        f'\nclaim\'s text: {input_sample["claim-fact"]}'
        f"news Article: {input_sample['article']}\n\n"
        f"stance: \n"
    )

    return [
        {
            "role": "system",
            "content": "You are a fact checking expert. Your task is to analyze the claim, text of the claim and the associated news article.",
        },
        {
            "role": "user",
            "content": prompt_string,
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"].lower()
    label = label.replace("label:", "")
    label_fixed = label.lower()
    label_fixed = label_fixed.split()[0]
    label_fixed = label_fixed.replace(".", "")

    return label_fixed
