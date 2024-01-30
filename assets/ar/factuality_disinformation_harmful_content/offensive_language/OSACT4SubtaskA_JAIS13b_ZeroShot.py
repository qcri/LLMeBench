from llmebench.datasets import OSACT4SubtaskADataset
from llmebench.models import FastChatModel
from llmebench.tasks import OffensiveTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "JAIS-13b",
        "description": "Locally hosted JAIS-13b-chat model using FastChat.",
        "scores": {"Macro-F1": ""},
    }


def config():
    return {
        "dataset": OSACT4SubtaskADataset,
        "task": OffensiveTask,
        "model": FastChatModel,
        "model_args": {
            "class_labels": ["OFF", "NOT_OFF"],
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    base_prompt = f'if the following Arabic sentence is offensive, just say "OFF", otherwise, say just "NOT_OFF" without explanation: \n {input_sample}'
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
    return out
