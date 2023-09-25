import re

from llmebench.datasets import CT22AttentionworthyDataset
from llmebench.models import LegacyOpenAIModel
from llmebench.tasks import AttentionworthyTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-35-turbo (version 0301)",
        "description": "GPT35 model hosted on Azure, using the Completion API. API version '2023-03-15-preview'.",
        "scores": {"Weighted-F1": "0.258"},
    }


def config():
    return {
        "dataset": CT22AttentionworthyDataset,
        "task": AttentionworthyTask,
        "model": LegacyOpenAIModel,
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
            "max_tries": 3,
        },
        "general_args": {"test_split": "ar"},
    }


def prompt(input_sample):
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": f"Classify the sentence by whether it should get the attention of policymakers. Answer by yes or no. If the predicted label is yes then classify the sentence into one of the following categories: asks question, blame authorities, calls for action, Harmful, contains advice, discusses action taken, discusses cure, or other.\n\ntext: {input_sample}label: ",
            }
        ],
    }


def post_process(response):
    label = response["choices"][0]["text"]

    label = label.lower().replace(" - ", ", ").replace(",", "").replace(".", "")
    label = re.sub("\s+", "_", label)
    if label.startswith("no"):
        label_fixed = "no_not_interesting"
    elif label == "yes_discusses_covid-19_vaccine_side_effects":
        label_fixed = "yes_discusses_cure"
    elif label == "yes_harmful":
        label_fixed = "harmful"
    elif label.startswith("yes"):
        label_fixed = label

    return label_fixed
