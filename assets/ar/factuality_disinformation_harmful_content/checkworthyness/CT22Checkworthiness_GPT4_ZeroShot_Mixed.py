import re

from llmebench.datasets import CT22CheckworthinessDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import CheckworthinessTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'.",
        "scores": {"F1 (POS)": "0.560"},
    }


def config():
    return {
        "dataset": CT22CheckworthinessDataset,
        "task": CheckworthinessTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["0", "1"],
            "max_tries": 30,
        },
        "general_args": {"test_split": "ar"},
    }


def prompt(input_sample):
    prompt_string = (
        f'هل تحتوي هذه "التغريدة" على ادعاء يستحق التحقق منه؟ أجب بـ checkworthy أو not_checkworthy\n\n'
        f"التغريدة: {input_sample}\n"
        f"التصنيف: \n"
    )
    return [
        {
            "role": "system",
            "content": "أنت خبير في تحليل وتصنيف التغريدات.",
        },
        {
            "role": "user",
            "content": prompt_string,
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"]

    label = label.replace("label:", "").strip().lower()

    if (
        "لا_يستحق_التحقق" in label
        or "لا يستحق التحقق" in label
        or "ليس يستحق التحقق" in label
        or "لا تستحق التحقق" in label
        or "no" in label
        or "لا" in label
        or "not" in label
    ):
        return "0"
    elif (
        "yes" in label
        or "نعم" in label
        or "يستحق التحقق" in label
        or "checkworthy" in label
    ):
        return "1"
    else:
        return None
