import re

from llmebench.datasets import ArProBinaryDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import ArProTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. 3 samples where chosen per test sample based on MaxMarginalRelevance for few shot learning.",
        "scores": {"Micro-F1": "0.592"},
    }


def config():
    return {
        "dataset": ArProBinaryDataset,
        "task": ArProTask,
        "model": OpenAIModel,
        "model_args": {
            "max_tries": 3,
        },
    }


def prompt(input_sample, examples):
    prompt_text = (
        f"Your task is to analyze the text and determine if it contains elements of propaganda.\n\n"
        f"Below you will find a few examples that can help you to understand:\n\n"
    )

    fs_prompt = few_shot_prompt(input_sample, prompt_text, examples)
    return [
        {
            "role": "system",
            "content": "You are an expert annotator.",
        },
        {
            "role": "user",
            "content": fs_prompt,
        },
    ]


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt
    for index, example in enumerate(examples):
        sent = example["input"]
        label = example["label"]
        out_prompt = (
            out_prompt
            + "Example "
            + str(index)
            + ":"
            + "\n"
            + "text: "
            + sent
            + "\nlabel: "
            + label
            + "\n\n"
        )

    out_prompt = (
        out_prompt
        + "Based on the instructions and examples above analyze the following 'text' and predict whether it contains the use of any propaganda technique. Answer only by true or false. Return only predicted label.\n\n"
    )
    out_prompt = out_prompt + "text: " + input_sample + "\nlabel: \n"

    return out_prompt


def post_process(response):
    input_label = response["choices"][0]["message"]["content"]
    pred_label = input_label.replace(".", "").strip().lower()

    pred_label = pred_label.replace("label:", "").strip()

    if "true" == pred_label:
        pred_label = "true"

    elif "false" == pred_label:
        pred_label = "false"
    else:
        pred_label = None

    return pred_label
