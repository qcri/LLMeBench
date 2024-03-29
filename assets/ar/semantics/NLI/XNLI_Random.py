from llmebench.datasets import XNLIDataset
from llmebench.models import RandomModel
from llmebench.tasks import TaskType, XNLITask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Random",
        "description": "Random Baseline.",
        "scores": {"Accuracy": "0.330"},
    }


def config():
    return {
        "dataset": XNLIDataset,
        "task": XNLITask,
        "model": RandomModel,
        "model_args": {
            "task_type": TaskType.Classification,
            "class_labels": ["contradiction", "entailment", "neutral"],
        },
        "general_args": {},
    }


def prompt(input_sample):
    return input_sample


def post_process(response):
    return response["random_response"]
