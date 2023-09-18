from llmebench.datasets import CT23SubjectivityDataset
from llmebench.models import RandomModel
from llmebench.tasks import SubjectivityTask, TaskType


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Random",
        "description": "Random Baseline.",
        "scores": {"Macro-F1": "0.496"},
    }


def config():
    return {
        "dataset": CT23SubjectivityDataset,
        "task": SubjectivityTask,
        "model": RandomModel,
        "model_args": {
            "task_type": TaskType.Classification,
            "class_labels": ["SUBJ", "OBJ"],
        },
        "general_args": {"test_split": "ar/dev"},
    }


def prompt(input_sample):
    return input_sample


def post_process(response):
    return response["random_response"]
