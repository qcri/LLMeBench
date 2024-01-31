from llmebench.datasets import ArapTweetDataset
from llmebench.models import RandomModel
from llmebench.tasks import ClassificationTask, TaskType


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Random",
        "description": "Random Baseline.",
        "scores": {"Macro-F1": "0.521"},
    }


def config():
    return {
        "dataset": ArapTweetDataset,
        "task": ClassificationTask,
        "model": RandomModel,
        "model_args": {
            "task_type": TaskType.Classification,
            "class_labels": ["Female", "Male"],
        },
        "general_args": {},
    }


def prompt(input_sample):
    return input_sample


def post_process(response):
    return response["random_response"]
