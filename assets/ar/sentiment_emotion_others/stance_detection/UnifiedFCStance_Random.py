from llmebench.datasets import UnifiedFCStanceDataset
from llmebench.models import RandomModel
from llmebench.tasks import StanceTask, TaskType


def config():
    return {
        "dataset": UnifiedFCStanceDataset,
        "dataset_args": {},
        "task": StanceTask,
        "task_args": {},
        "model": RandomModel,
        "model_args": {
            "task_type": TaskType.Classification,
            "class_labels": ["agree", "disagree", "discuss", "unrelated"],
        },
        "general_args": {},
    }


def prompt(input_sample):
    return input_sample


def post_process(response):
    return response["random_response"]
