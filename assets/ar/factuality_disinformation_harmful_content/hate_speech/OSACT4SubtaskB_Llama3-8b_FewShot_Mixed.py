import random

from llmebench.datasets import OSACT4SubtaskBDataset
from llmebench.models import AzureModel
from llmebench.tasks import HateSpeechTask


def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "Llama-3.1-8B-Instruct",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054).",
    }


def config():
    return {
        "dataset": OSACT4SubtaskBDataset,
        "task": HateSpeechTask,
        "model": AzureModel,
        "model_args": {
            "class_labels": ["HS", "NOT_HS"],
            "max_tries": 3,
        },
    }


def few_shot_prompt(input_sample, examples):
    base_prompt = (
        "أجب فقط بـ 'hate_speech' إذا كانت خطاب كراهية و'not_hate_speech' إذا لم تكن خطاب كراهية من التغريدات التالية. "
        "إليك بعض الأمثلة لتوجيهك:\n\n"
    )

    for index, example in enumerate(examples):
        label = "hate_speech" if example["label"] == "HS" else "not_hate_speech"
        base_prompt += (
            f"مثال {index + 1}:\n"
            f"التغريدة: '{example['input']}'\n"
            f"التصنيف: {label}\n\n"
        )
    base_prompt += (
        f"الآن، قم بتقييم التغريدة الجديدة التالية:\nالتغريدة: '{input_sample}'\n"
        f"التصنيف (يرجى الرد فقط بـ 'كراهية' أو 'ليست كراهية'):"
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
    print(response)
    if "output" in response:
        label = response["output"].strip()
        label = label.replace("<s>", "").replace("</s>", "").lower()
    else:
        print("Response .. " + str(response))
        return "NOT_HS"  # Default to "NOT_HS" when unsure

    if (
        "not hate speech" in label
        or "not_hs" in label
        or "لا كراهية" in label
        or "لا" in label
        or "ليست" in label
        or "ليس" in label
        or "no" in label
    ):
        return "NOT_HS"
    elif "hate speech" in label or "hs" in label or "كراهية" in label or "hate_speech":
        return "HS"
    else:
        print("No clear label found.")
        return random.choice(["HS", "NOT_HS"])
