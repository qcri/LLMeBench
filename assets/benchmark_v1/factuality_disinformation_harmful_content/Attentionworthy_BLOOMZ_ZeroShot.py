import os
import random

from arabic_llm_benchmark.datasets import AttentionworthyDataset
from arabic_llm_benchmark.models import BLOOMPetalModel
from arabic_llm_benchmark.tasks import AttentionworthyTask


random.seed(1333)


def config():
    return {
        "dataset": AttentionworthyDataset,
        "dataset_args": {},
        "task": AttentionworthyTask,
        "task_args": {},
        "model": BLOOMPetalModel,
        "model_args": {
            "api_url": os.environ["API_URL"],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/attentionworthy/CT22_arabic_1D_attentionworthy_test_gold.tsv"
        },
    }


def prompt(input_sample):
    prompt = (
        f'Predict whether a tweet should get the attention of policy makers. Use the following categories: yes_discusses_action_taken, harmful, yes_discusses_cure, yes_asks_question, no_not_interesting, yes_other, yes_blame_authorities, yes_contains_advice, yes_calls_for_action\n\n'
        f"tweet: {input_sample}\n"
        f"label: \n"
    )
    return {
            "prompt": prompt,
        }


def post_process(response):
    label = response["outputs"].lower().replace("<s>", "").replace("</s>", "").strip()
    label_fixed = None

    if label.startswith("no"):
        label_fixed = "no_not_interesting"
    elif label == "yes_discusses_covid-19_vaccine_side_effects":
        label_fixed = "yes_discusses_cure"
    elif label == "yes_harmful":
        label_fixed = "harmful"
    elif label.startswith("yes"):
        label_fixed = label

    if label_fixed == None:
        print("Issue with label! " + label)

    return label_fixed
