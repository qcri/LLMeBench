from llmebench.datasets import ANERcorpDataset
from llmebench.models import RandomModel
from llmebench.tasks import NERTask, TaskType


def config():
    return {
        "dataset": ANERcorpDataset,
        "dataset_args": {},
        "task": NERTask,
        "task_args": {},
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
