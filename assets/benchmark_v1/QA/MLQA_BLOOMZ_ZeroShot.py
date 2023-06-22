import os

from arabic_llm_benchmark.datasets import MLQADataset
from arabic_llm_benchmark.models import BLOOMPetalModel
from arabic_llm_benchmark.tasks import QATask


def config():
    return {
        "dataset": MLQADataset,
        "dataset_args": {},
        "task": QATask,
        "task_args": {},
        "model": BLOOMPetalModel,
        "model_args": {
            "api_url": os.environ["API_URL"],
            "max_tries": 5,
        },
        "general_args": {"data_path": "data/QA/mlqa/test-context-ar-question-ar.json"},
    }


def prompt(input_sample):
    return {
        "prompt": "Your task is to answer arabic questions based on a given context. Your answers should be extracted from the context." + "\n"
        + f"Context: {input_sample['context']}\n"
        + f"Question: {input_sample['question']}\n"
        + "Answer: "
    }


def post_process(response):
    return response["outputs"]
