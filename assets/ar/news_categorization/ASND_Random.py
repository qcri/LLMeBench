from llmebench.datasets import ASNDDataset
from llmebench.models import RandomModel
from llmebench.tasks import NewsCategorizationTask, TaskType


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Random",
        "description": "Random Baseline.",
        "scores": {"Macro-F1": "0.048"},
    }


def config():
    return {
        "dataset": ASNDDataset,
        "task": NewsCategorizationTask,
        "model": RandomModel,
        "model_args": {
            "task_type": TaskType.Classification,
            "class_labels": [
                "crime-war-conflict",
                "spiritual",
                "health",
                "politics",
                "human-rights-press-freedom",
                "education",
                "business-and-economy",
                "art-and-entertainment",
                "others",
                "science-and-technology",
                "sports",
                "environment",
            ],
        },
        "general_args": {},
    }


def prompt(input_sample):
    return input_sample


def post_process(response):
    return response["random_response"]
