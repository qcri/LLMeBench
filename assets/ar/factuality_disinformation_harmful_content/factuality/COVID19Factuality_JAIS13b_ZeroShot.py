from llmebench.datasets import COVID19FactualityDataset
from llmebench.models import FastChatModel
from llmebench.tasks import FactualityTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "JAIS-13b",
        "description": "Locally hosted JAIS-13b-chat model using FastChat.",
        "scores": {"Macro-F1": ""},
    }


def config():
    return {
        "dataset": COVID19FactualityDataset,
        "task": FactualityTask,
        "model": FastChatModel,
        "model_args": {
            "class_labels": ["yes", "no"],
            "max_tries": 30,
        },
    }


def prompt(input_sample):
    base_prompt = (
        f'Annotate the "tweet" into one of the following categories: correct or incorrect\n\n'
        f"tweet: {input_sample}\n"
        f"label: \n"
    )
    return [
        {
            "role": "user",
            "content": base_prompt,
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"]

    if label.startswith("I am unable to verify".lower()) or label.startswith(
        "I am unable to categorize".lower()
    ):
        label_fixed = None
    elif "label: incorrect" in label or "incorrect" in label:
        label_fixed = "no"
    elif "label: correct" in label or "correct" in label:
        label_fixed = "yes"
    else:
        label_fixed = None

    return label_fixed
