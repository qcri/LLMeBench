from llmebench.datasets import MGBWordsDataset
from llmebench.models import RandomModel
from llmebench.tasks import NERTask, TaskType


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Random",
        "description": "Random Baseline.",
        "scores": {"Macro-F1": "0.009"},
    }


def config():
    return {
        "dataset": MGBWordsDataset,
        "task": NERTask,
        "model": RandomModel,
        "model_args": {
            "task_type": TaskType.SequenceLabeling,
            "class_labels": [
                "B-PERS",
                "I-PERS",
                "B-LOC",
                "I-LOC",
                "B-ORG",
                "I-ORG",
                "B-MISC",
                "I-MISC",
                "O",
            ],
        },
        "general_args": {},
    }


def prompt(input_sample):
    return input_sample


def post_process(response):
    return response["random_response"]
