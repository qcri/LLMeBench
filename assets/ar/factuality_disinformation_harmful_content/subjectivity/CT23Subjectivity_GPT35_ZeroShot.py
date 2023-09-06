import os

from llmebench.datasets import CT23SubjectivityDataset
from llmebench.models import LegacyOpenAIModel
from llmebench.tasks import SubjectivityTask


def config():
    return {
        "dataset": CT23SubjectivityDataset,
        "dataset_args": {},
        "task": SubjectivityTask,
        "task_args": {},
        "model": LegacyOpenAIModel,
        "model_args": {
            "class_labels": ["SUBJ", "OBJ"],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/subjectivity/dev_ar.tsv"
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
    label = response["choices"][0]["text"]
    if label == "Objective" or label == "Objective.":
        label_fixed = "OBJ"
    elif label == "Subjective" or label == "Subjective.":
        label_fixed = "SUBJ"

    return label_fixed
