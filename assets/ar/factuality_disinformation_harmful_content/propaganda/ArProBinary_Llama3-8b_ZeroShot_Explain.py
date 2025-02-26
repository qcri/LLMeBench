import random
import re

from llmebench.datasets import ArProBinaryDataset
from llmebench.models import AzureModel
from llmebench.tasks import ArProTask


random.seed(1333)


def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "Llama-3.1-8B-Instruct",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054).",
    }


def config():
    return {
        "dataset": ArProBinaryDataset,
        "task": ArProTask,
        "model": AzureModel,
        "model_args": {
            "max_tries": 5,
        },
    }


def prompt(input_sample):
    prompt_text = (
        f"Classify the given text as 'propagandistic' or 'non-propagandistic' and explain your reasoning. Return only the label and Arabic explanation in the following format:\nLabel: predicted label\nExplanation: Arabic explanation of predicted label.\n"
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

    if "i can't" in label or "i cannot" in label or "لا أستطيع" in label:
        return None

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
