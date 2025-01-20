import random

from llmebench.datasets import CT22HarmfulDataset
from llmebench.models import FastChatModel
from llmebench.tasks import HarmfulDetectionTask


def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "jais-13b-chat",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054).",
    }


def config():
    return {
        "dataset": CT22HarmfulDataset,
        "task": HarmfulDetectionTask,
        "model": FastChatModel,
        "model_args": {
            "class_labels": ["0", "1"],
            "max_tries": 30,
        },
        "general_args": {"test_split": "ar"},
    }


def prompt(input_sample):
    return [
        {
            "role": "user",
            "content": (
                "صنف التغريدة التالية على أنها 'harmful' أو 'not_harmful'. قدم التصنيف فقط.\n\n"
                + "تغريدة: "
                + input_sample
                + "التصنيف: "
            ),
        }
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"].lower()

    if "غير" in label or "ليس" in label or "not" in label:
        return "0"
    elif (
        "ضار" in label
        or "مس" in label
        or "مؤ" in label
        or "harm" in label
        or "مض" in label
    ):
        return "1"

    else:
        return None
