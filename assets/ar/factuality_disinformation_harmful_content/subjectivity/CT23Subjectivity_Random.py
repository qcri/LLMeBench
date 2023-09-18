from llmebench.datasets import CT23SubjectivityDataset
from llmebench.models import RandomModel
from llmebench.tasks import SubjectivityTask, TaskType


def config():
    return {
        "dataset": CT23SubjectivityDataset,
        "dataset_args": {},
        "task": SubjectivityTask,
        "task_args": {},
        "model": RandomModel,
        "model_args": {
            "task_type": TaskType.Classification,
            "class_labels": ["SUBJ", "OBJ"],
        },
        "general_args": {"test_split": "ar/dev"},
    }


def prompt(input_sample):
    return input_sample


def post_process(response):
    return response["random_response"]
