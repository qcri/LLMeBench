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
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    prompt_text = (
        f"Your task is to analyze the text and determine if it contains elements of propaganda. Based on the instructions, analyze the following 'text' and predict whether it contains the use of any propaganda technique. Answer only by true or false. Return only predicted label.\n\n"
        f"text: {input_sample}\n"
        f"label: \n"
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

    if "true" in label or "label: 1" in label or "yes" in label:
        pred_label = "true"
    elif (
        "false" in label
        or "label: 0" in label
        or "label: no" in label
        or "non" in label
        or "not" in label
    ):
        pred_label = "false"
    else:
        print("label problem!! " + label)
        pred_label = None

    return pred_label
