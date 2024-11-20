import re

from llmebench.datasets import OSACT4SubtaskADataset
from llmebench.models import AzureModel
from llmebench.tasks import OffensiveTask


def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "Llama-3.1-8B-Instruct",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054).",
    }


def config():
    return {
        "dataset": OSACT4SubtaskADataset,
        "task": OffensiveTask,
        "model": AzureModel,
        "model_args": {
            "class_labels": ["OFF", "NOT_OFF"],
            "max_tries": 3,
        },
    }


# def few_shot_prompt(input_sample, examples):
#     base_prompt = "Review the content provided and classify it as 'OFF' for Offensive or 'NOT_OFF' for Not Offensive. Here are some examples to guide your classification:\n\n"
#     for index, example in enumerate(examples):
#         label = "NOT_OFF" if example["label"] == "NOT_OFF" else "OFF"
#         base_prompt += f"Example {index + 1}: Content - '{example['input']}' -> Classification: {label}\n"
#     base_prompt += "\nNow classify the new content:\nContent: '" + input_sample + "'\nClassification:\n"
#     return base_prompt


def few_shot_prompt(input_sample, examples):
    base_prompt = f"Please assess the following sentence to determine if it contains any offensive content. If the sentence contains any degree of offense, respond with 'OFF'. If the sentence does not contain any offensive, insulting, or harmful material, respond with 'NOT_OFF'. Do not provide any additional explanation.Here are some examples:\n\n"
    for index, example in enumerate(examples):
        label = "NOT_OFF" if example["label"] == "NOT_OFF" else "OFF"
        base_prompt += f"Example {index + 1}: Content - '{example['input']}' -> Classification: {label}\n"
    base_prompt += (
        "\nNow classify the new content:\nContent: '"
        + input_sample
        + "'\nClassification:\n"
    )
    return base_prompt


def prompt(input_sample, examples):
    return [
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, examples),
        }
    ]


def post_process(response):
    label = response["output"].strip().lower()
    label = re.sub(r"<s>|</s>", "", label)  # Removing potential formatting characters
    label = label.lower()

    if "not" in label or "غير" in label or "ليس" in label:
        return "NOT_OFF"
    elif "is" in label or "مسيء" in label or "off" in label:
        return "OFF"
    else:
        return None
