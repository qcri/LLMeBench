import os
import random
import re

from arabic_llm_benchmark.datasets import AttentionworthyDataset
from arabic_llm_benchmark.models import GPTChatCompletionModel
from arabic_llm_benchmark.tasks import AttentionworthyTask


random.seed(1333)


def config():
    return {
        "dataset": AttentionworthyDataset,
        "dataset_args": {},
        "task": AttentionworthyTask,
        "task_args": {},
        "model": GPTChatCompletionModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
            "class_labels": [
                "yes_discusses_action_taken",
                "harmful",
                "yes_discusses_cure",
                "yes_asks_question",
                "no_not_interesting",
                "yes_other",
                "yes_blame_authorities",
                "yes_contains_advice",
                "yes_calls_for_action",
            ],
            "max_tries": 30,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/attentionworthy/CT22_arabic_1D_attentionworthy_test_gold.tsv",
            "fewshot": {
                "train_data_path": "data/factuality_disinformation_harmful_content/attentionworthy/CT22_arabic_1D_attentionworthy_train.tsv",
            },
        },
    }


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n"
    for example in examples:
        out_prompt = (
            out_prompt
            + "tweet: "
            + example["input"]
            + "\nlabel: "
            + example["label"]
            + "\n\n"
        )

    # Append the sentence we want the model to predict for but leave the Label blank
    out_prompt = out_prompt + "tweet: " + input_sample + "\nlabel: \n"

    return out_prompt


def prompt(input_sample, examples):
    base_prompt = f'Annotate the following "tweet" into one of the following categories: yes_discusses_action_taken, harmful, yes_discusses_cure, yes_asks_question, no_not_interesting, yes_other, yes_blame_authorities, yes_contains_advice, yes_calls_for_action. Provide only label.\n\n'
    return [
        {
            "role": "system",
            "content": "You are social media expert. You can annotate important tweets that require attention from journalists, fact-checker, and government entities.",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"]
    label_list = config()["model_args"]["class_labels"]
    label = (
        label.lower()
        .replace(" - ", ", ")
        .replace(",", "")
        .replace(".", "")
        .replace("label:", "")
    )
    label = label.strip()
    # label = re.sub("\s+", "_", label)
    if label.startswith("no"):
        label_fixed = "no_not_interesting"
    elif label == "yes_discusses_covid-19_vaccine_side_effects":
        label_fixed = "yes_discusses_cure"
    elif label == "yes_harmful":
        label_fixed = "harmful"
    elif label.startswith("yes"):
        label_fixed = label
    elif "label: yes_blame_authoritie" in label:
        label_fixed = "yes_blame_authoritie"
    elif "label: yes_discusses_action_taken" in label:
        label_fixed = "yes_discusses_action_taken"
    elif "label: harmful" in label:
        label_fixed = "harmful"
    elif "label: yes_discusses_cure" in label:
        label_fixed = "yes_discusses_cure"
    elif "label: yes_asks_question" in label:
        label_fixed = "yes_asks_question"
    elif "label: no_not_interesting" in label:
        label_fixed = "no_not_interesting"
    elif "label: yes_other" in label:
        label_fixed = "yes_other"
    elif "label: yes_blame_authorities" in label:
        label_fixed = "yes_blame_authorities"
    elif "label: yes_contains_advice" in label:
        label_fixed = "yes_contains_advice"
    elif "label: yes_calls_for_action" in label:
        label_fixed = "yes_calls_for_action"
    elif label in label_list:
        label_fixed = label
    else:
        label_fixed = None

    return label_fixed
