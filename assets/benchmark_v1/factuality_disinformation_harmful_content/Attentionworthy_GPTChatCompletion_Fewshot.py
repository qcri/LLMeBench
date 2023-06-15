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
        # Found chatgpt confused when using 0 and 1 in the prompt
        label = "no" if example["label"] == "0" else "yes"
        out_prompt = (
            out_prompt + "tweet: " + example["input"] + "\nlabel: " + label + "\n\n"
        )

    # Append the sentence we want the model to predict for but leave the Label blank
    out_prompt = out_prompt + "tweet: " + input_sample + "\nlabel: \n"

    # print("=========== FS Prompt =============\n")
    # print(out_prompt)

    return out_prompt


def prompt(input_sample, examples):
    base_prompt = f'Annotate "tweet" into one of the following categories: yes_discusses_action_taken, harmful, yes_discusses_cure, yes_asks_question, no_not_interesting, yes_other, yes_blame_authorities, yes_contains_advice, yes_calls_for_action\n\n'
    return [
        {
            "role": "system",
            "content": "You are social media expert. You can annotate important tweets and require attention from journalists, fact-checker, and government entities.",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"]

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

    return label_fixed
