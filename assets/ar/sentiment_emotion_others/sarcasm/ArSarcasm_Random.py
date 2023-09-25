from llmebench.datasets import ArSarcasmDataset
from llmebench.models import RandomModel
from llmebench.tasks import SarcasmTask, TaskType


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Random",
        "description": "Random Baseline.",
        "scores": {"F1 (POS)": "0.240"},
    }


def config():
    return {
        "dataset": ArSarcasmDataset,
        "task": SarcasmTask,
        "model": RandomModel,
        "model_args": {
            "task_type": TaskType.Classification,
            "class_labels": ["TRUE", "FALSE"],
        },
        "general_args": {},
    }


def prompt(input_sample):
    return input_sample


def post_process(response):
    return response["random_response"]
