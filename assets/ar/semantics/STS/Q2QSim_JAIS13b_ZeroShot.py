from llmebench.datasets import STSQ2QDataset
from llmebench.models import FastChatModel
from llmebench.tasks import Q2QSimDetectionTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Jais-13b-chat",
        "description": "Locally hosted Jais-13b-chat model using FastChat.",
    }


def config():
    return {
        "dataset": STSQ2QDataset,
        "task": Q2QSimDetectionTask,
        "model": FastChatModel,
        "model_args": {
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    q1, q2 = input_sample.split("\t")
    input_sample = q1 + "\t" + q2
    base_prompt = f"Are the following two questions semantically similar (i.e., asking for similar information)? The output should be exactly in form yes or no.\n\n{input_sample}"

    return [
        {
            "role": "user",
            "content": base_prompt,
        },
    ]


def post_process(response):
    input_label = response["choices"][0]["message"]["content"]
    input_label = input_label.replace(".", "").strip().lower()
    pred_label = ""

    if "yes" in input_label or "label: 1" in input_label:
        pred_label = "1"
    if (
        input_label == "no"
        or input_label.startswith("no,")
        or "label: 0" in input_label
        or "label: no" in input_label
        or "not semantically similar" in input_label
    ):
        pred_label = "0"

    if pred_label == "":
        pred_label = None

    return pred_label
