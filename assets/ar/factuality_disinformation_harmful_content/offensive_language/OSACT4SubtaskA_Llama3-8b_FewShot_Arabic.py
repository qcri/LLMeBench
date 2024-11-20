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


def few_shot_prompt(input_sample, examples):
    base_prompt = f"يرجى تقييم الجملة التالية لتحديد ما إذا كانت تحتوي على محتوى مسيء. إذا كانت الجملة تحتوي على أي درجة من الإساءة، قم بالرد بـ 'مسيء'. إذا لم تحتوي الجملة على أي محتوى مسيء أو مهين أو ضار، قم بالرد بـ 'غير مسيء'.\n"
    for index, example in enumerate(examples):
        label = "غير مسيء" if example["label"] == "NOT_OFF" else "مسيء"
        base_prompt += (
            f"مثال {index + 1}:\n"
            f"الجملة: '{example['input']}'\n"
            f"التصنيف: {label}\n\n"
        )
    base_prompt += (
        f"الآن، قم بتقييم الجملة الجديدة التالية:\nالجملة: '{input_sample}'\n"
        f" التصنيف:"
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

    if "غير" in label or "not" in label or "ليس" in label or "ليست" in label:
        return "NOT_OFF"
    elif (
        "مسيء" in label or "offensive" in label or "المساءاة" in label or "off" in label
    ):
        return "OFF"
    else:
        print("No clear label found:", label)
        return None
