from llmebench.datasets import ArSASDataset
from llmebench.models import FastChatModel
from llmebench.tasks import SentimentTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Llama-2-13b-chat-hf",
        "description": "Locally hosted Llama-2-13b-chat hf model using FastChat. Poor performance is expected, since Llama 2 is not explicitly trained with Arabic data.",
        "scores": {"Macro-F1": "0.106"},
    }


def config():
    return {
        "dataset": ArSASDataset,
        "task": SentimentTask,
        "model": FastChatModel,
    }


def prompt(input_sample):
    return [
        {
            "role": "system",
            "content": "You are an AI assistant that helps people find information.",
        },
        {
            "role": "user",
            "content": f'Classify the sentiment of the following sentence as "Positive", "Negative", "Neutral" or "Mixed". Output only the label and nothing else.\nSentence: {input_sample}\nLabel: ',
        },
    ]


def post_process(response):
    out = response["choices"][0]["message"]["content"]
    out = out.strip().lower()

    if "i apologize" in out:
        return None

    j = out.find("label:")
    if j > 0:
        out = out[j + len("label:") :]
    else:
        j = out.find(" is:\n\n")
        if j > 0:
            out = out[j + len(" is:\n\n") :]
    out = out.strip().title()
    return out
