import random

from llmebench.datasets import PADTDataset
from llmebench.models import RandomModel
from llmebench.tasks import ArabicParsingTask, TaskType


def config():
    return {
        "dataset": PADTDataset,
        "dataset_args": {},
        "task": ArabicParsingTask,
        "task_args": {},
        "model": RandomModel,
        "model_args": {"task_type": TaskType.Other},
        "general_args": {},
    }


def prompt(input_sample):
    return input_sample


def post_process(response):
    tokens = response["random_response"].split(" ")
    random_labels = [str(idx) for idx in range(len(tokens) + 1)]
    random_response = {
        str(idx + 1): random.choice(random_labels) for idx in range(len(tokens))
    }

    return random_response
