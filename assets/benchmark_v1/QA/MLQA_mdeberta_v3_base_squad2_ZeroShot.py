import os

from llmebench.datasets import MLQADataset
from llmebench.models import HuggingFaceInferenceAPIModel, HuggingFaceTaskTypes
from llmebench.tasks import QATask


def config():
    return {
        "dataset": MLQADataset,
        "dataset_args": {},
        "task": QATask,
        "task_args": {},
        "model": HuggingFaceInferenceAPIModel,
        "model_args": {
            "task_type": HuggingFaceTaskTypes.Question_Answering,
            "inference_api_url": "https://api-inference.huggingface.co/models/timpal0l/mdeberta-v3-base-squad2",
            "api_token": os.environ["HUGGINGFACE_API_TOKEN"],
            "max_tries": 5,
        },
        "general_args": {
            "data_path": "data/QA/MLQA/test/test-context-ar-question-ar.json"
        },
    }


def prompt(input_sample):
    return {
        "inputs": {
            "context": input_sample["context"],
            "question": input_sample["question"],
        }
    }


def post_process(response):
    return response["answer"].strip()
