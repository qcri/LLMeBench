from llmebench.datasets import HuggingFaceDataset
from llmebench.models import FastChatModel
from llmebench.tasks import ClassificationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "togethercomputer/RedPajama-INCITE-Chat-3B-v1",
        "description": (
            "Locally hosted RedPajama-INCITE-Chat 3b parameters model using FastChat."
        ),
        "scores": {"Accuracy": "0.681"},
    }


def config():
    return {
        "dataset": HuggingFaceDataset,
        "dataset_args": {
            "huggingface_dataset_name": "sst2",
            "column_mapping": {
                "input": "sentence",
                "label": "label",
                "input_id": "idx",
            },
        },
        "task": ClassificationTask,
        "model": FastChatModel,
        "general_args": {"custom_test_split": "validation"},
    }


def prompt(input_sample):
    prompt_string = (
        f"You are tasked with analyzing the sentiment of the given sentence. "
        f"Please read it carefully and determine whether the sentiment expressed is positive or negative. Provide only label.\n\n"
        f"sentence: {input_sample.strip()}\n"
        f"label:\n"
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a data annotation expert specializing in sentiment analysis."
            ),
        },
        {"role": "user", "content": prompt_string},
    ]


def post_process(response):
    response = response["choices"][0]["message"]["content"].lower()
    response = response.replace(".", "")
    response = response.strip()

    if response.endswith("positive"):
        return 1
    elif response.endswith("negative"):
        return 0
    elif "positive" in response and "negative" not in response:
        return 1
    elif "negative" in response and "positive" not in response:
        return 0
    else:
        return None
