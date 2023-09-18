from llmebench.datasets import XQuADDataset
from llmebench.models import PetalsModel
from llmebench.tasks import QATask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "bloomz-176b (8bit quantized)",
        "description": "Locally hosted BLOOMZ 176b model (8 bit quantized version) using the Petals.",
        "scores": {"F1": "0.367"},
    }


def config():
    return {
        "dataset": XQuADDataset,
        "dataset_args": {},
        "task": QATask,
        "task_args": {},
        "model": PetalsModel,
        "model_args": {
            "max_tries": 5,
        },
    }


def prompt(input_sample):
    return {
        "prompt": "Your task is to answer arabic questions based on a given context. Your answers should be extracted from the context."
        + "\n"
        + f"Context: {input_sample['context']}\n"
        + f"Question: {input_sample['question']}\n"
        + "Answer: "
    }


def post_process(response):
    return response["outputs"]
