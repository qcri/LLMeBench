from llmebench.datasets import TyDiQADataset
from llmebench.models import PetalsModel
from llmebench.tasks import QATask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "bloomz-176b (8bit quantized)",
        "description": "Locally hosted BLOOMZ 176b model (8 bit quantized version) using the Petals.",
        "scores": {"F1": "0.456"},
    }


def config():
    return {
        "dataset": TyDiQADataset,
        "task": QATask,
        "model": PetalsModel,
        "model_args": {
            "max_tries": 5,
        },
        "general_args": {"test_split": "dev"},
    }


def prompt(input_sample):
    return {
        "prompt": (
            "Your task is to answer arabic questions based on a given context. Your answers should be extracted from the context."
            + "\n"
            + f"Context: {input_sample['context']}\n"
            + f"Question: {input_sample['question']}\n"
            + "Answer: "
        )
    }


def post_process(response):
    return response["outputs"]
