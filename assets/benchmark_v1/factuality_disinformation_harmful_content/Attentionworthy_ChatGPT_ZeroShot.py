import os
import re

from arabic_llm_benchmark.datasets import SubjectivityDataset
from arabic_llm_benchmark.models import GPTModel, RandomGPTModel
from arabic_llm_benchmark.tasks import AttentionworthyTask


def config():
    return {
        "dataset": SubjectivityDataset,
        "dataset_args": {},
        "task": AttentionworthyTask,
        "task_args": {"test": "useless"},
        "model": RandomGPTModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": "gpt",
            "class_labels": ["SUBJ", "OBJ"],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/attentionworthy/CT22_arabic_1D_attentionworthy_dev.tsv"
        },
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
    label = response["response"]["choices"][0]["message"]["content"]

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
