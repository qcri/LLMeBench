import random
import re

from llmebench.datasets import FinePropBinaryDataset
from llmebench.models import AzureModel
from llmebench.tasks import ArProTask


random.seed(1333)


def metadata():
    return {
        "author": "Maram Hasanain and Md Arid Hasan and Mohamed Bayan Kmainasi and Elisa Sartori and Ali Ezzat Shahroor and Giovanni Da San Martino and Firoj Alam",
        "model": "Llama3-8b",
        "description": "https://arxiv.org/abs/2502.16550",
        "scores": {},
    }


def config():
    return {
        "dataset": FinePropBinaryDataset,
        "task": ArProTask,
        "model": AzureModel,
        "model_args": {
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    prompt_text = (
        f"Classify the given text as 'propagandistic' or 'non-propagandistic' and explain your reasoning. Return only the label and English explanation in the following format:\nLabel: predicted label\nExplanation: English explanation of predicted label.\n"
        f"text: {input_sample}\n"
        f"Label: \n"
        f"Explanation: \n"
    )

    return [
        {
            "role": "user",
            "content": prompt_text,
        },
    ]


def post_process(response):
    label = response["output"].strip().lower()
    label = label.replace("<s>", "").replace("</s>", "")
    label = label.lower()

    if (
        "i can't" in label
        or "i cannot" in label
        or "لا أستطيع" in label
        or "i don't" in label
        or "i'm ready to help" in label
    ):
        return None

    if "propagandistic: " in label:
        pred_label = "true"
    else:
        splits = label.split("explanation:")
        label = (
            splits[0]
            .replace("explanation", "")
            .replace("label", "")
            .replace(":", "")
            .strip()
        )
        explain = (
            splits[1]
            .replace("explanation", "")
            .replace("label", "")
            .replace(":", "")
            .strip()
        )

        if (
            "true" in label
            or "نعم" in label
            or "label: 1" in label
            or "yes" in label
            or label == "propagandistic"
            or label == "دعائي"
            or "يعتبر النص من النوع الدعائي" in explain
        ):
            pred_label = "true"
        elif (
            "false" in label
            or "ليس" in label
            or "غير" in label
            or "label: 0" in label
            or "label: no" in label
            or "non" in label
            or "not" in label
            or "لا يحتوي" in explain
            or "لا يوجد" in explain
            or "ليس منشورًا دعائيًا" in explain
        ):
            pred_label = "false"
        else:
            print("label problem!! " + label)
            pred_label = None

    return pred_label
