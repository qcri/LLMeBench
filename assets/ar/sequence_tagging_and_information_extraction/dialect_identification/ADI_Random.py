from llmebench.datasets import ADIDataset
from llmebench.models import RandomModel
from llmebench.tasks import DialectIDTask, TaskType


def config():
    return {
        "dataset": ADIDataset,
        "dataset_args": {},
        "task": DialectIDTask,
        "task_args": {},
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
