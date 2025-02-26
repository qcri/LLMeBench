import re

from llmebench.datasets import CT22HarmfulDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import HarmfulDetectionTask


def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "GPT-4o-2024-05-22",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054).",
    }


def config():
    return {
        "dataset": CT22HarmfulDataset,
        "task": HarmfulDetectionTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["0", "1"],
            "max_tries": 30,
        },
        "general_args": {"test_split": "ar"},
    }


def prompt(input_sample):
    prompt_string = (
        f"هل قد تتسبب هذه التغريدة بأي ضرر أو أذى لشخص أو مؤسسة؟ إن كانت كذلك، أجب بـ 'harmful'. أجب بـ 'not harmful' إن لم تكن كذلك. قدم فقط التصنيف.\n\n"
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
    label_fixed = None
    label = label.replace("label:", "")
    label = label.replace("التصنيف:", "")
    label = label.strip().lower()

    if (
        "غير ضارة" in label
        or "ليست ضارة" in label
        or "غير ضاره" in label
        or "غير" in label
        or "not" in label
        or "ليست" in label
        or "لا" in label
        or "not harmful" in label
        or label.startswith("no")
        or "safe" in label
        or "not_harmful" in label
    ):
        return "0"
    elif "ضارة" in label or "harmful" in label or "نعم" in label or "yes" in label:
        return "1"

    return label_fixed
