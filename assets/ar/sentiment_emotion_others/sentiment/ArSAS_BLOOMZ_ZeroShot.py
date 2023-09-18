from llmebench.datasets import ArSASDataset
from llmebench.models import PetalsModel
from llmebench.tasks import SentimentTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "bloomz-176b (8bit quantized)",
        "description": "Locally hosted BLOOMZ 176b model (8 bit quantized version) using the Petals.",
        "scores": {"Macro-F1": "0.251"},
    }


def config():
    return {
        "dataset": ArSASDataset,
        "task": SentimentTask,
        "model": PetalsModel,
        "model_args": {
            "class_labels": ["Positive", "Negative", "Neutral", "Mixed"],
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    return {
        "prompt": "Choose only one sentiment between: Positive, Negative, Neutral, or Mixed for this Arabic sentence. Provide only label.\n\n"
        + "sentence: "
        + input_sample
        + "label: "
    }


def post_process(response):
    label = response["outputs"].strip()
    label = label.replace("<s>", "")
    label = label.replace("</s>", "")
    return label
