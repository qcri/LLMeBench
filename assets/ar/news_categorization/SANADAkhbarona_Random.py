from llmebench.datasets import SANADAkhbaronaDataset
from llmebench.models import RandomModel
from llmebench.tasks import NewsCategorizationTask, TaskType


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Random",
        "description": "Random Baseline.",
        "scores": {"Accuracy": "0.142"},
    }


def config():
    return {
        "dataset": SANADAkhbaronaDataset,
        "task": NewsCategorizationTask,
        "model": RandomModel,
        "model_args": {
            "task_type": TaskType.Classification,
            "class_labels": [
                "politics",
                "religion",
                "medical",
                "sports",
                "tech",
                "finance",
                "culture",
            ],
        },
        "general_args": {},
    }


def prompt(input_sample):
    return input_sample


def post_process(response):
    return response["random_response"]
