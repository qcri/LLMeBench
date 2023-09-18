from llmebench.datasets import WikiNewsPOSDataset
from llmebench.models import RandomModel
from llmebench.tasks import ArabicPOSTask, TaskType


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Random",
        "description": "Random Baseline.",
        "scores": {"Accuracy": "0.019"},
    }


def config():
    return {
        "dataset": WikiNewsPOSDataset,
        "dataset_args": {},
        "task": ArabicPOSTask,
        "task_args": {},
        "model": RandomModel,
        "model_args": {
            "task_type": TaskType.SequenceLabeling,
            "class_labels": [
                "ABBREV",
                "ADJ",
                "ADJ/CONJ",
                "ADJ/DET",
                "ADJ/NUM",
                "ADV",
                "CASE",
                "CONJ",
                "DET",
                "FOREIGN",
                "FUT_PART",
                "NOUN",
                "NOUN/DET",
                "NSUFF",
                "NSUFF/ADJ",
                "NSUFF/DET",
                "NSUFF/NOUN",
                "NUM",
                "PART",
                "PART/CONJ",
                "PART/NOUN",
                "PART/PART",
                "PART/PREP",
                "PREP",
                "PRON",
                "PUNC",
                "V",
            ],
        },
        "general_args": {},
    }


def prompt(input_sample):
    return input_sample


def post_process(response):
    return response["random_response"]
