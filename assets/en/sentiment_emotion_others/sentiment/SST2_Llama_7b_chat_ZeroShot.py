from llmebench.datasets import HuggingFaceDataset
from llmebench.models import FastChatModel
from llmebench.tasks import SentimentTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "...",
        "description": "...",
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
        "task": SentimentTask,
        "model": FastChatModel,
        "general_args": {"custom_test_split": "validation"},
    }


def prompt(input_sample):
    return [
        {
            "role": "system",
            "content": "You are an AI assistant that helps people find information.",
        },
        {
            "role": "user",
            "content": f'Classify the sentiment of the following sentence as "Positive" or "Negative". Output only the label and nothing else.\nSentence: {input_sample}\nLabel: ',
        },
    ]


def post_process(response):
    out = response["choices"][0]["message"]["content"]
    out = out.strip().lower()
    j = out.find("label:")
    if j > 0:
        out = out[j + len("label:") :]
    out = out.strip().lower()

    if out == "positive":
        return 1
    elif out == "negative":
        return 0
    else:
        return None
