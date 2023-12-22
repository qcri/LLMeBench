from llmebench.datasets import ArSASDataset
from llmebench.models import FastChatModel
from llmebench.tasks import SentimentTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "JAIS-13b",
        "description": "Locally hosted JAIS-13b-chat model using FastChat.",
        "scores": {"Macro-F1": ""},
    }


def config():
    return {
        "dataset": ArSASDataset,
        "task": SentimentTask,
        "model": FastChatModel,
    }


def prompt(input_sample):
    base_prompt = (
        f'Classify the sentiment of the following sentence as "Positive", "Negative", "Neutral" or "Mixed". Output only the label and nothing else.\n'
        f"Sentence: {input_sample}\n"
        f"Label: "
    )

    return [
        {
            "role": "user",
            "content": base_prompt,
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
