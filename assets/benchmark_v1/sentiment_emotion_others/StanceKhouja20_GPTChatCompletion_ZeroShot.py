import os
import random
import re

from arabic_llm_benchmark.datasets import StanceKhouja20Dataset
from arabic_llm_benchmark.models import GPTChatCompletionModel
from arabic_llm_benchmark.tasks import StanceKhouja20Task


random.seed(1333)


def config():
    return {
        "dataset": StanceKhouja20Dataset,
        "dataset_args": {},
        "task": StanceKhouja20Task,
        "task_args": {},
        "model": GPTChatCompletionModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
            "class_labels": ["agree", "disagree"],
            "max_tries": 30,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/factuality_stance_khouja/stance/test.csv"
        },
    }


def prompt(input_sample):
    prompt_string = (
        f"Based on your analysis, determine the stance of the 'first sentence' with respect to the 'second sentence'. The possible stances could be 'agree', or 'disagree'."
        f"\n\n"
        f"first sentence: {input_sample['sentence_1']}"
        f"second sentence: {input_sample['sentence_2']}"
        f"stance: \n"
    )

    return [
        {
            "role": "system",
            "content": "You are a fact checking expert. Your task is to analyze the stance between two sentences.",
        },
        {
            "role": "user",
            "content": prompt_string,
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"].lower()
    label = label.replace("label:", "").strip()
    label_fixed = label.replace("stance:", "").strip()

    if label_fixed.startswith("the two sentences are unrelated"):
        label_fixed = None
    elif "the stance could be considered 'disagree'" in label_fixed:
        label_fixed = "disagree"

    return label_fixed
