import random
import re

from llmebench.datasets import SpamDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import SpamTask


random.seed(1333)


def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "GPT-4o-2024-05-22",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054).",
    }


def config():
    return {
        "dataset": SpamDataset,
        "task": SpamTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["__label__ADS", "__label__NOTADS"],
            "max_tries": 3,
        },
    }


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n"
    out_prompt = out_prompt + "Here are some examples:\n\n"
    for index, example in enumerate(examples):
        label = "spam" if example["label"] == "__label__ADS" else "not spam"

        out_prompt = (
            out_prompt
            + "Example "
            + str(index)
            + ":"
            + "\n"
            + "tweet: "
            + example["input"]
            + "\nlabel: "
            + label
            + "\n\n"
        )

    # Append the tweet we want the model to predict for but leave the Label blank
    out_prompt = out_prompt + "tweet: " + input_sample + "\nlabel: \n"

    return out_prompt


def prompt(input_sample, examples):
    base_prompt = (
        f"If the following tweet can be classified as spam or contains an advertisemnt, write 'spam' without explnanation, otherwise write 'not spam' without explanantion.\n\n"
        f"Provide only labels as a list of string.\n"
    )

    return [
        {
            "role": "system",
            "content": "You are an expert social media content analyst.",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def post_process(response):
    out = response["choices"][0]["message"]["content"]
    label = out.replace("التصنيف:", "").strip().lower()
    label = label.replace("label:", "").strip().lower()

    # print("label", label)
    if "لا أستطيع" in label or "I cannot" in label:
        return None
    if (
        "ليست" in label
        or "not" in label
        or "no" in label
        or "ليس" in label
        or "notads" in label
    ):
        return "__label__NOTADS"
    elif (
        "نعم" in label
        or "إعلان" in label
        or "spam" in label
        or "مزعج" in label
        or "اعلان" in label
        or "مرغوب" in label
        or "غير" in label
        or "__ads" in label
    ):
        return "__label__ADS"
    else:
        return None
