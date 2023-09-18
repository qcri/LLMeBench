from llmebench.datasets import CT22CheckworthinessDataset
from llmebench.models import RandomModel
from llmebench.tasks import CheckworthinessTask, TaskType


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Random",
        "description": "Random Baseline.",
        "scores": {"F1 (POS)": "0.100"},
    }


def config():
    return {
        "dataset": CT22CheckworthinessDataset,
        "task": CheckworthinessTask,
        "model": RandomModel,
        "model_args": {
            "task_type": TaskType.Classification,
            "class_labels": ["0", "1"],
        },
        "general_args": {"test_split": "tr"},
    }


def prompt(input_sample):
    return input_sample


def post_process(response):
    return response["random_response"]
