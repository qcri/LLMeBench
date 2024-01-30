from llmebench.datasets import SpamDataset
from llmebench.models import FastChatModel
from llmebench.tasks import SpamTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "JAIS-13b",
        "description": "Locally hosted JAIS-13b-chat model using FastChat.",
        "scores": {"Macro-F1": ""},
    }


def config():
    return {
        "dataset": SpamDataset,
        "task": SpamTask,
        "model": FastChatModel,
        "model_args": {
            "class_labels": ["__label__ADS", "__label__NOTADS"],
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    base_prompt = f"If the following sentence can be classified as spam or contains an advertisemnt, write '__label__ADS' without explnanation, otherwise write '__label__NOTADS' without explanantion.\n {input_sample}\n"
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
    if "NOTADS" in out:
        out = "__label__NOTADS"
    elif "ADS" in out:
        out = "__label__ADS"
    return out
