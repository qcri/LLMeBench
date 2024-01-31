from llmebench.datasets import OSACT4SubtaskBDataset
from llmebench.models import FastChatModel
from llmebench.tasks import HateSpeechTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Jais-13b-chat",
        "description": "Locally hosted Jais-13b-chat model using FastChat.",
    }


def config():
    return {
        "dataset": OSACT4SubtaskBDataset,
        "task": HateSpeechTask,
        "model": FastChatModel,
        "model_args": {
            "class_labels": ["HS", "NOT_HS"],
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    base_prompt = f'Given the following tweet, label it as "HS" or "NOT_HS" based on the content of the tweet: \n {input_sample}'
    return [
        {
            "role": "user",
            "content": base_prompt,
        },
    ]


def post_process(response):
    out = response["choices"][0]["message"]["content"]
    j = out.find(".")
    if j > 0:
        out = out[0:j]

    if "not_hate_speech" in out or "no_hate_speech" in out or "NOT_HS" == out:
        out = "NOT_HS"
    elif "hate_speech" in out or "HS" == out:
        out = "HS"
    else:
        out = None
    return out
