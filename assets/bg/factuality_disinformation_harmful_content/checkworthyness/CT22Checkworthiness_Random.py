from llmebench.datasets import CT22CheckworthinessDataset
from llmebench.models import RandomModel
from llmebench.tasks import CheckworthinessTask, TaskType


def config():
    return {
        "dataset": CT22CheckworthinessDataset,
        "dataset_args": {},
        "task": CheckworthinessTask,
        "task_args": {},
        "model": RandomModel,
        "model_args": {
            "task_type": TaskType.Classification,
            "class_labels": ["0", "1"],
        },
        "general_args": {"test_split": "bg"},
    }


def prompt(input_sample):
    return input_sample


def post_process(response):
    return response["random_response"]
