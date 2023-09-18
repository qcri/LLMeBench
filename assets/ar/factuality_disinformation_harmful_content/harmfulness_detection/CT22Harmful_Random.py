from llmebench.datasets import CT22HarmfulDataset
from llmebench.models import RandomModel
from llmebench.tasks import HarmfulDetectionTask, TaskType


def config():
    return {
        "dataset": CT22HarmfulDataset,
        "dataset_args": {},
        "task": HarmfulDetectionTask,
        "task_args": {},
        "model": RandomModel,
        "model_args": {
            "task_type": TaskType.Classification,
            "class_labels": ["0", "1"],
        },
        "general_args": {"test_split": "ar"},
    }


def prompt(input_sample):
    return input_sample


def post_process(response):
    return response["random_response"]
