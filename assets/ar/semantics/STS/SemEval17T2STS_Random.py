from llmebench.datasets import SemEval17T2STSDataset
from llmebench.models import RandomModel
from llmebench.tasks import STSTask, TaskType


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Random",
        "description": "Random Baseline.",
        "scores": {"PC": "-0.136"},
    }


def config():
    return {
        "dataset": SemEval17T2STSDataset,
        "task": STSTask,
        "model": RandomModel,
        "model_args": {"task_type": TaskType.Regression, "score_range": (0, 5)},
        "general_args": {},
    }


def prompt(input_sample):
    return input_sample


def post_process(response):
    return response["random_response"]
