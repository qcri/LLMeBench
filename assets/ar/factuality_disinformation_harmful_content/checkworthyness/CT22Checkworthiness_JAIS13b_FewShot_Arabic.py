import re

from llmebench.datasets import CT22CheckworthinessDataset
from llmebench.models import FastChatModel
from llmebench.tasks import CheckworthinessTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "LLama3-8b",
        "description": "Deployed on Azure.",
        "scores": {"F1 (POS)": "0.554"},
    }


def config():
    return {
        "dataset": CT22CheckworthinessDataset,
        "task": CheckworthinessTask,
        "model": FastChatModel,
        "model_args": {
            "class_labels": ["0", "1"],
            "max_tries": 30,
        },
        "general_args": {"test_split": "ar", "fewshot": {"train_split": "ar"}},
    }


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n"
    out_prompt = out_prompt + "هذه بعض الأمثلة:\n\n"
    for index, example in enumerate(examples):
        label = "غير جديرة بالتحقق" if example["label"] == "0" else "جديرة بالتحقق"

        out_prompt = (
            out_prompt
            + "مثال "
            + str(index)
            + ":"
            + "\n"
            + "التغريدة: "
            + example["input"]
            + "\التصنيف: "
            + label
            + "\n\n"
        )

    out_prompt = out_prompt + "التغريدة: " + input_sample + "\التصنيف: \n"

    return out_prompt


def prompt(input_sample, examples):
    base_prompt = ' صنف التغريدة الى "جديرة بالتحقق" أو "غير جديرة بالتحقق". قم بتقديم التصنيف فقط لأخر تغريدة، لا تقدم أي تبرير إضافي.:\n'
    return [
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"]

    label = label.replace("label:", "").strip()

    label = label.lower()
    if (
        "لا" in label
        or "غير" in label
        or "no" in label
        or "ليس" in label
        or "ليست" in label
    ):
        return "0"
    return "1"
    return label
