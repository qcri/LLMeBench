from llmebench.datasets import ADIDataset
from llmebench.models import RandomModel
from llmebench.tasks import DialectIDTask, TaskType


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Random",
        "description": "Random Baseline.",
        "scores": {"Macro-F1": "0.062"},
    }


def config():
    return {
        "dataset": ADIDataset,
        "task": DialectIDTask,
        "model": RandomModel,
        "model_args": {
            "task_type": TaskType.Classification,
            "class_labels": [
                "egy",
                "ira",
                "jor",
                "ksa",
                "kuw",
                "leb",
                "lib",
                "mor",
                "msa",
                "pal",
                "qat",
                "sud",
                "syr",
                "uae",
                "YEM",
            ],
        },
        "general_args": {},
    }


def prompt(input_sample):
    return input_sample


def post_process(response):
    return response["random_response"]
