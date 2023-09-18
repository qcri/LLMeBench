from llmebench.datasets import QADIDataset
from llmebench.models import RandomModel
from llmebench.tasks import DialectIDTask, TaskType


def config():
    return {
        "dataset": QADIDataset,
        "dataset_args": {},
        "task": DialectIDTask,
        "task_args": {},
        "model": RandomModel,
        "model_args": {
            "task_type": TaskType.Classification,
            "class_labels": [
                "EG",
                "DZ",
                "SD",
                "YE",
                "SY",
                "TN",
                "AE",
                "JO",
                "LY",
                "PS",
                "OM",
                "LB",
                "KW",
                "QA",
                "BH",
                "MSA",
                "SA",
                "IQ",
                "MA",
            ],
        },
        "general_args": {},
    }


def prompt(input_sample):
    return input_sample


def post_process(response):
    return response["random_response"]
