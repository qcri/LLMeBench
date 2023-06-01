import os

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
            "ignore_cache": True,
        },
        "general_args": {
            "data_path": "tasks/factuality_disinformation_harmful_content/subjectivity/data/dev_ar.tsv"
        },
    }


def prompt(input_sample):
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": f"Classify the sentence by whether it should get the attention of policymakers. Answer by yes or no. If the predicted label is yes then classify the sentence into one of the following categories: asks question, blame authorities, calls for action, Harmful, contains advice, discusses action taken, discusses cure, or other.\n\ntext: {input_sample}\nlabel:",
            }
        ],
    }


def post_process(response):
    label = response["response"]["choices"][0]["message"]["content"]
    return label
