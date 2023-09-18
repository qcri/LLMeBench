from llmebench.datasets import STSQ2QDataset
from llmebench.models import HuggingFaceInferenceAPIModel, HuggingFaceTaskTypes
from llmebench.tasks import Q2QSimDetectionTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "intfloat/multilingual-e5-small",
        "description": "Sample HuggingFace Inference API asset for sentence similarity.",
        "scores": {"Micro-F1": "ar/semantics/STS/Q2QSim"},
    }


def config():
    return {
        "dataset": STSQ2QDataset,
        "dataset_args": {},
        "task": Q2QSimDetectionTask,
        "task_args": {},
        "model": HuggingFaceInferenceAPIModel,
        "model_args": {
            "task_type": HuggingFaceTaskTypes.Sentence_Similarity,
            "inference_api_url": "https://api-inference.huggingface.co/models/intfloat/multilingual-e5-small",
            "max_tries": 5,
        },
    }


def prompt(input_sample):
    q1, q2 = input_sample.split("\t")

    return {"inputs": {"source_sentence": q1, "sentences": [q2]}}


def post_process(response):
    if response[0] > 0.7:
        pred_label = "1"
    elif response[0] < 0.3:
        pred_label = "0"
    else:
        pred_label = None

    return pred_label
