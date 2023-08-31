import os

from llmebench.datasets import SubjectivityDataset
from llmebench.models import BLOOMPetalModel
from llmebench.tasks import SubjectivityTask


def config():
    return {
        "dataset": SubjectivityDataset,
        "dataset_args": {},
        "task": SubjectivityTask,
        "task_args": {},
        "model": BLOOMPetalModel,
        "model_args": {
            "api_url": os.environ["API_URL"],
            "class_labels": ["SUBJ", "OBJ"],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/subjectivity/dev_ar.tsv"
        },
    }


def prompt(input_sample):
    return {
        "prompt": f"Classify the sentence as Subjective or Objective. Provide only label.\n\ntext: {input_sample}\nlabel: ",
    }


def post_process(response):
    label = response["outputs"].strip().lower()
    label = label.replace("<s>", "")
    label = label.replace("</s>", "")

    if label == "objective" or label == "objective.":
        label_fixed = "OBJ"
    elif label == "subjective" or label == "subjective.":
        label_fixed = "SUBJ"

    return label_fixed
