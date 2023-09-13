from llmebench.datasets import MLQADataset
from llmebench.models import PetalsModel
from llmebench.tasks import QATask


def config():
    return {
        "dataset": MLQADataset,
        "dataset_args": {},
        "task": QATask,
        "task_args": {},
        "model": PetalsModel,
        "model_args": {
            "max_tries": 5,
        },
        "general_args": {
            "data_path": "data/QA/MLQA/test/test-context-ar-question-ar.json"
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
