from llmebench.datasets import AdultDataset
from llmebench.models import RandomModel
from llmebench.tasks import AdultTask, TaskType


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Random",
        "description": "Random Baseline.",
        "scores": {"Macro-F1": "0.421"},
    }


def config():
    return {
        "dataset": AdultDataset,
        "task": AdultTask,
        "model": RandomModel,
        "model_args": {
            "task_type": TaskType.Classification,
            "class_labels": ["ADULT", "NOT_ADULT"],
        },
        "general_args": {},
    }


def prompt(input_sample):
    return input_sample


def post_process(response):
    return response["random_response"]
