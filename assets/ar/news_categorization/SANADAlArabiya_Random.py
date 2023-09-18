from llmebench.datasets import SANADAlArabiyaDataset
from llmebench.models import RandomModel
from llmebench.tasks import NewsCategorizationTask, TaskType


def config():
    return {
        "dataset": SANADAlArabiyaDataset,
        "dataset_args": {},
        "task": NewsCategorizationTask,
        "task_args": {},
        "model": RandomModel,
        "model_args": {
            "task_type": TaskType.Classification,
            "class_labels": [
                "politics",
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
