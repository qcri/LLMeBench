import random

from llmebench.datasets import ArSASDataset
from llmebench.models import VLLMModel
from llmebench.tasks import SentimentTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Fanar 7B",
        "description": "Locally hosted Fanar 7B model using the VLLM.",
        # "scores": {"Macro-F1": "0.251"}, # needs to be updated after applying post-processing
    }


def config():
    return {
        "dataset": ArSASDataset,
        "task": SentimentTask,
        "model": VLLMModel,
        "model_args": {
            "class_labels": ["Positive", "Negative", "Neutral", "Mixed"],
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    return [
        {
            "role": "user",
            "content": (
                "Choose only one sentiment between: Positive, Negative, Neutral, or Mixed for this Arabic sentence. Provide only label.\n\n"
                + "sentence: "
                + input_sample
                + "label: "
            ),
        }
    ]


def post_process(response):
    if "messages" in response:
        if "content" in response["messages"]:
            label = response["messages"]["content"].strip()
            label = label.replace("<s>", "")
            label = label.replace("</s>", "")
    elif "content" in response["messages"]:
        label = response["messages"]["content"].strip()
        label = label.replace("<s>", "")
        label = label.replace("</s>", "")
    else:
        random.seed(1234)
        labels = config()["model_args"]["class_labels"]
        label = random.choice(labels)
    return label
