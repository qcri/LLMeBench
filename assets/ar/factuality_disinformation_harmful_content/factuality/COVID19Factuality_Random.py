from llmebench.datasets import COVID19FactualityDataset
from llmebench.models import RandomModel
from llmebench.tasks import FactualityTask, TaskType


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Random",
        "description": "Random Baseline.",
        "scores": {"Weighted-F1": "0.582"},
    }


def config():
    return {
        "dataset": COVID19FactualityDataset,
        "dataset_args": {},
        "task": FactualityTask,
        "task_args": {},
        "model": RandomModel,
        "model_args": {
            "task_type": TaskType.Classification,
            "class_labels": ["yes", "no"],
        },
        "general_args": {},
    }


def prompt(input_sample):
    return input_sample


def post_process(response):
    return response["random_response"]
