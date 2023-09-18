from llmebench.datasets import QCRIDialectalArabicPOSDataset
from llmebench.models import RandomModel
from llmebench.tasks import ArabicPOSTask, TaskType


def config():
    return {
        "dataset": QCRIDialectalArabicPOSDataset,
        "dataset_args": {},
        "task": ArabicPOSTask,
        "task_args": {},
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
