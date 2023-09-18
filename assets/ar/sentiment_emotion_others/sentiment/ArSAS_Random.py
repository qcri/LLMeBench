from llmebench.datasets import ArSASDataset
from llmebench.models import RandomModel
from llmebench.tasks import SentimentTask, TaskType


def config():
    return {
        "dataset": ArSASDataset,
        "dataset_args": {},
        "task": SentimentTask,
        "task_args": {},
        "model": RandomModel,
        "model_args": {
            "task_type": TaskType.Classification,
            "class_labels": ["Positive", "Negative", "Neutral", "Mixed"],
        },
        "general_args": {},
    }


def prompt(input_sample):
    return input_sample


def post_process(response):
    return response["random_response"]
