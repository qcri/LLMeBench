import os

from arabic_llm_benchmark.datasets import SubjectivityDataset
from arabic_llm_benchmark.models import GPTModel, RandomGPTModel
from arabic_llm_benchmark.tasks import SubjectivityTask


def config():
    return {
        "dataset": SubjectivityDataset,
        "dataset_args": {},
        "task": SubjectivityTask,
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
            "data_path": "data/factuality_disinformation_harmful_content/subjectivity/data/dev_ar.tsv"
        },
    }


def prompt(input_sample):
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": f"Classify the sentence as Subjective or Objective. Provide only label.\n\ntext: {input_sample}label: ",
            }
        ],
    }


def post_process(response):
    label = response["response"]["choices"][0]["message"]["content"]
    if (label == "Objective" or label == "Objective."):
        label_fixed = "OBJ"
    elif (label == "Subjective" or label == "Subjective."):
        label_fixed = "SUBJ"

    return label_fixed
