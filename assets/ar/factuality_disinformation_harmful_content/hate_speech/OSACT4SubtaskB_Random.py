from llmebench.datasets import OSACT4SubtaskBDataset
from llmebench.models import RandomModel
from llmebench.tasks import HateSpeechTask, TaskType


def config():
    return {
        "dataset": OSACT4SubtaskBDataset,
        "dataset_args": {},
        "task": HateSpeechTask,
        "task_args": {},
        "model": RandomModel,
        "model_args": {
            "task_type": TaskType.Classification,
            "class_labels": ["HS", "NOT_HS"],
        },
        "general_args": {},
    }


def prompt(input_sample):
    return input_sample


def post_process(response):
    return response["random_response"]