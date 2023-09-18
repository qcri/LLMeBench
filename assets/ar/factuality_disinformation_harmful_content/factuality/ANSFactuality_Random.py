from llmebench.datasets import ANSFactualityDataset
from llmebench.models import RandomModel
from llmebench.tasks import FactualityTask, TaskType


def config():
    return {
        "dataset": ANSFactualityDataset,
        "dataset_args": {},
        "task": FactualityTask,
        "task_args": {},
        "model": RandomModel,
        "model_args": {
            "task_type": TaskType.Classification,
            "class_labels": ["true", "false"],
        },
        "general_args": {},
    }


def prompt(input_sample):
    return input_sample


def post_process(response):
    return response["random_response"]
