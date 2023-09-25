from llmebench.datasets import QCRIDialectalArabicPOSDataset
from llmebench.models import RandomModel
from llmebench.tasks import ArabicPOSTask, TaskType


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Random",
        "description": "Random Baseline.",
        "scores": {"Accuracy": "0.026"},
    }


def config():
    return {
        "dataset": QCRIDialectalArabicPOSDataset,
        "task": ArabicPOSTask,
        "model": RandomModel,
        "model_args": {
            "task_type": TaskType.SequenceLabeling,
            "class_labels": [
                "ADJ",
                "ADV",
                "CASE",
                "CONJ",
                "DET",
                "EMOT",
                "FOREIGN",
                "FUT_PART",
                "HASH",
                "MENTION",
                "NEG_PART",
                "NOUN",
                "NSUFF",
                "NUM",
                "PART",
                "PREP",
                "PROG_PART",
                "PRON",
                "PUNC",
                "URL",
                "V",
            ],
        },
        "general_args": {},
    }


def prompt(input_sample):
    return input_sample


def post_process(response):
    return response["random_response"]
