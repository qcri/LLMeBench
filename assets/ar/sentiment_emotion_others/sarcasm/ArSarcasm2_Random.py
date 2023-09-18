from llmebench.datasets import ArSarcasm2Dataset
from llmebench.models import RandomModel
from llmebench.tasks import SarcasmTask, TaskType


def config():
    return {
        "dataset": ArSarcasm2Dataset,
        "dataset_args": {},
        "task": SarcasmTask,
        "task_args": {},
        "model": RandomModel,
        "model_args": {
            "task_type": TaskType.Classification,
            "class_labels": ["TRUE", "FALSE"],
        },
        "general_args": {},
    }


def prompt(input_sample):
    return input_sample


def post_process(response):
    return response["random_response"]