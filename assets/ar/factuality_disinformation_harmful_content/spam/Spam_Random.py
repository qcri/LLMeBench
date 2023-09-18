from llmebench.datasets import SpamDataset
from llmebench.models import RandomModel
from llmebench.tasks import SpamTask, TaskType


def config():
    return {
        "dataset": SpamDataset,
        "dataset_args": {},
        "task": SpamTask,
        "task_args": {},
        "model": RandomModel,
        "model_args": {
            "task_type": TaskType.Classification,
            "class_labels": ["__label__ADS", "__label__NOTADS"],
        },
        "general_args": {},
    }


def prompt(input_sample):
    return input_sample


def post_process(response):
    return response["random_response"]
