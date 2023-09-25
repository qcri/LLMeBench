from llmebench.datasets import ANSStanceDataset
from llmebench.models import RandomModel
from llmebench.tasks import StanceTask, TaskType


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Random",
        "description": "Random Baseline.",
        "scores": {"Macro-F1": "0.281"},
    }


def config():
    return {
        "dataset": ANSStanceDataset,
        "task": StanceTask,
        "model": RandomModel,
        "model_args": {
            "task_type": TaskType.Classification,
            "class_labels": ["agree", "disagree"],
        },
        "general_args": {},
    }


def prompt(input_sample):
    return input_sample


def post_process(response):
    return response["random_response"]
