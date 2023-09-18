from llmebench.datasets import OSACT4SubtaskADataset
from llmebench.models import RandomModel
from llmebench.tasks import OffensiveTask, TaskType


def config():
    return {
        "dataset": OSACT4SubtaskADataset,
        "dataset_args": {},
        "task": OffensiveTask,
        "task_args": {},
        "model": RandomModel,
        "model_args": {
            "task_type": TaskType.Classification,
            "class_labels": ["OFF", "NOT_OFF"],
        },
        "general_args": {},
    }


def prompt(input_sample):
    return input_sample


def post_process(response):
    return response["random_response"]
