import random

from llmebench.datasets import PADTDataset
from llmebench.models import RandomModel
from llmebench.tasks import ArabicParsingTask, TaskType


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Random",
        "description": "Random Baseline.",
        "scores": {"UAS": "0.001"},
    }


def config():
    return {
        "dataset": PADTDataset,
        "task": ArabicParsingTask,
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
