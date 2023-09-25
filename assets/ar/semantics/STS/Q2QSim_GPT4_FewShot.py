import re

from llmebench.datasets import STSQ2QDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import Q2QSimDetectionTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. 3 samples where chosen per test sample based on MaxMarginalRelevance for few shot learning.",
        "scores": {"Micro-F1": "0.935"},
    }


def config():
    return {
        "dataset": STSQ2QDataset,
        "task": Q2QSimDetectionTask,
        "model": OpenAIModel,
        "model_args": {
            "max_tries": 3,
        },
    }


def prompt(input_sample, examples):
    q1, q2 = input_sample.split("\t")
    # prompt = f"Are the following two questions semantically similar (i.e., asking for similar information)? The output should be exactly in form yes or no.\n\n{input_sample}"
    base_prompt = f"Are the two questions below semantically similar (i.e., asking for similar information)? The output should be exactly in form yes or no.\n\n"
    prompt = few_shot_prompt(q1, q2, base_prompt, examples)

    return [
        {
            "role": "system",
            "content": "You are an AI assistant that helps people find information.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


def few_shot_prompt(q1, q2, base_prompt, examples):
    out_prompt = base_prompt

    for example in examples:
        ex_q1, ex_q2 = example["input"].split("\t")
        label = "no" if example["label"] == "0" else "yes"

        out_prompt = (
            out_prompt
            + "Q1: "
            + ex_q1
            + "\nQ2: "
            + ex_q2
            + "\n"
            + "label: "
            + label
            + "\n\n"
        )

    # Append the sentence we want the model to predict for but leave the Label blank
    out_prompt = out_prompt + "Q1: " + q1 + "\nQ2: " + q2 + "\nlabel: \n"

    # print("=========== FS Prompt =============\n")
    # print(out_prompt)

    return out_prompt


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
