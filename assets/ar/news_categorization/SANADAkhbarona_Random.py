from llmebench.datasets import SANADAkhbaronaDataset
from llmebench.models import RandomModel
from llmebench.tasks import NewsCategorizationTask, TaskType


def config():
    return {
        "dataset": SANADAkhbaronaDataset,
        "dataset_args": {},
        "task": NewsCategorizationTask,
        "task_args": {},
        "model": RandomModel,
        "model_args": {
            "task_type": TaskType.Classification,
            "class_labels": [
                "politics",
                "religion",
                "medical",
                "sports",
                "tech",
                "finance",
                "culture",
            ],
        },
        "general_args": {},
    }


def prompt(input_sample):
    return input_sample


def post_process(response):
    return response["random_response"]
