from llmebench.datasets import AdultDataset
from llmebench.models import RandomModel
from llmebench.tasks import AdultTask, TaskType


def config():
    return {
        "dataset": AdultDataset,
        "dataset_args": {},
        "task": AdultTask,
        "task_args": {},
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
