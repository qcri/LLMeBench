import random

from llmebench.datasets import ARCDDataset
from llmebench.models import RandomModel
from llmebench.tasks import QATask, TaskType


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Random",
        "description": "Random Baseline.",
        "scores": {"F1": "0.085"},
    }


def config():
    return {
        "dataset": ARCDDataset,
        "task": QATask,
        "model": RandomModel,
        "model_args": {"task_type": TaskType.QuestionAnswering},
        "general_args": {},
    }


def prompt(input_sample):
    return input_sample


def post_process(response):
    tokens = response["random_response"]["context"].split(" ")

    start_idx = random.choice(range(len(tokens)))
    answer_length = random.choice(range(len(tokens) - start_idx))

    return " ".join(tokens[start_idx : start_idx + answer_length])
