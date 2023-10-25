from llmebench.datasets import SST2Dataset
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
        "dataset": SST2Dataset,
        "task": SentimentTask,
        "model": FastChatModel,
        "general_args": {"test_split": "dev"},
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
        return "1"
    elif out == "negative":
        return "0"
    else:
        return None
