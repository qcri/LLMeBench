from llmebench.datasets import XGLUEPOSDataset
from llmebench.models import RandomModel
from llmebench.tasks import ArabicPOSTask, TaskType


def config():
    return {
        "dataset": XGLUEPOSDataset,
        "dataset_args": {},
        "task": ArabicPOSTask,
        "task_args": {},
        "model": RandomModel,
        "model_args": {
            "task_type": TaskType.SequenceLabeling,
            "class_labels": [
                "ADJ",
                "ADP",
                "ADV",
                "AUX",
                "CCONJ",
                "DET",
                "INTJ",
                "NOUN",
                "NUM",
                "PART",
                "PRON",
                "PROPN",
                "PUNCT",
                "SYM",
                "VERB",
                "X",
            ],
        },
        "general_args": {},
    }


def prompt(input_sample):
    return input_sample


def post_process(response):
    return response["random_response"]
