from llmebench.datasets import AdultDataset
from llmebench.models import FastChatModel
from llmebench.tasks import AdultTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "JAIS-13b",
        "description": "Locally hosted JAIS-13b-chat model using FastChat.",
        "scores": {"Macro-F1": ""},
    }


def config():
    return {
        "dataset": AdultDataset,
        "task": AdultTask,
        "model": FastChatModel,
        "model_args": {
            "class_labels": ["ADULT", "NOT_ADULT"],
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    base_prompt = (
        f'Given the following tweet, label it as "ADULT" or "NOT_ADULT" based on the content of the tweet.\n\n'
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
    out = response["choices"][0]["message"]["content"].replace("label: ", "")
    j = out.find(".")
    if j > 0:
        out = out[0:j]
    return out
