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
    ref_s = input_sample.split("\t")[0]
    claim = input_sample.split("\t")[1]
    prompt_string = (
        f"Given a reference sentence and a claim, predict whether the claim agrees or disagrees with the reference sentence. Reply only using 'agree', 'disagree', or use 'other' if the sentence and claim are unrelated."
        f"\n\n"
        f"reference sentence: {ref_s}"
        f"\nclaim: {claim}"
        f"\nlabel: \n"
    )

    return [
        {
            "role": "system",
            "content": "You are a fact checking expert.",
        },
        {
            "role": "user",
            "content": prompt_string,
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"].lower()
    label = label.replace("label:", "").strip()
    #label_fixed = label.replace("stance:", "").strip()

    label_fixed = None
    #print(label)

    if "unrelated" in label or "other" in label:
        label_fixed = "other"
    elif "disagree" in label:
        label_fixed = "disagree"
    elif label == "agree":
        label_fixed = "agree"


    return label_fixed
