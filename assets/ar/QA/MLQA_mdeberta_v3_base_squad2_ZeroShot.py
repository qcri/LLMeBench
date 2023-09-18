from llmebench.datasets import MLQADataset
from llmebench.models import HuggingFaceInferenceAPIModel, HuggingFaceTaskTypes
from llmebench.tasks import QATask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "timpal0l/mdeberta-v3-base-squad2",
        "description": "Sample HuggingFace Inference API asset for question answering.",
        "scores": {"F1": "ar/QA/MLQA"},
    }


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
            "max_tries": 5,
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
