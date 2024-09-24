from llmebench.datasets import CT22AttentionworthyDataset
from llmebench.models import FastChatModel
from llmebench.tasks import AttentionworthyTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "LLama3-8b",
        "description": "Deployed on Azure.",
        "scores": {"Weighted-F1": "0.412"},
    }


def config():
    return {
        "dataset": CT22AttentionworthyDataset,
        "task": AttentionworthyTask,
        "model": FastChatModel,
        "model_args": {
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
            "max_tries": 100,
        },
        "general_args": {"test_split": "ar", "fewshot": {"train_split": "ar"}},
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
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


import re


def post_process(response):
    label = response["choices"][0]["message"]["content"]

    label = label.replace("label:", "").strip()

    label = label.replace("<s>", "").replace("</s>", "")
    label_fixed = (
        label.lower().strip()
    )  # Convert to lowercase and strip leading/trailing whitespace

    # Update conditions to match labels without surrounding whitespace
    if label_fixed.startswith("no"):
        label_fixed = "no_not_interesting"
    elif "yes_discusses_covid-19_vaccine_side_effects" in label_fixed:
        label_fixed = "yes_discusses_cure"
    elif "yes_harmful" in label_fixed:
        label_fixed = "harmful"
    elif label_fixed.startswith("yes"):
        label_fixed = (
            label_fixed.strip()
        )  # Keep the original label if it starts with "yes"
    elif "yes_blame_authoritie" in label_fixed:
        label_fixed = "yes_blame_authoritie"
    elif "yes_discusses_action_taken" in label_fixed:
        label_fixed = "yes_discusses_action_taken"
    elif "harmful" in label_fixed:
        label_fixed = "harmful"
    elif "yes_discusses_cure" in label_fixed:
        label_fixed = "yes_discusses_cure"
    elif "yes_asks_question" in label_fixed:
        label_fixed = "yes_asks_question"
    elif "no_not_interesting" in label_fixed:
        label_fixed = "no_not_interesting"
    elif "yes_other" in label_fixed:
        label_fixed = "yes_other"
    elif "yes_blame_authorities" in label_fixed:
        label_fixed = "yes_blame_authorities"
    elif "yes_contains_advice" in label_fixed:
        label_fixed = "yes_contains_advice"
    elif "yes_calls_for_action" in label_fixed:
        label_fixed = "yes_calls_for_action"
    else:
        label_fixed = None

    return label_fixed
