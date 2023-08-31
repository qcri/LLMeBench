import os
import re

from llmebench.datasets import Q2QSimDataset
from llmebench.models import GPTChatCompletionModel
from llmebench.tasks import Q2QSimDetectionTask


def config():
    return {
        "dataset": Q2QSimDataset,
        "dataset_args": {},
        "task": Q2QSimDetectionTask,
        "task_args": {},
        "model": GPTChatCompletionModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/STS/nsurl-2019-task8/test.tsv",
            "fewshot": {
                "train_data_path": "data/STS/nsurl-2019-task8/train.tsv",
            },
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
