import re

from llmebench.datasets import CT22CheckworthinessDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import CheckworthinessTask


def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "GPT-4o-2024-05-22",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054).",
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
        "general_args": {"test_split": "ar", "fewshot": {"train_split": "ar"}},
    }


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n"
    out_prompt = out_prompt + "اليك بعض الامثلة:\n\n"
    for index, example in enumerate(examples):
        label = "لا" if example["label"] == "0" else "نعم"

        out_prompt = (
            out_prompt
            + "مثال "
            + str(index)
            + ":"
            + "\n"
            + "التغريدة: "
            + example["input"]
            + "التصنيف: "
            + label
            + "\n\n"
        )

    # Append the sentence we want the model to predict for but leave the Label blank
    out_prompt = out_prompt + "التغريدة: " + input_sample + "\التصنيف: \n"

    return out_prompt


def prompt(input_sample, examples):
    base_prompt = f'هل تحتوي هذه "التغريدة" على ادعاء يستحق التحقق منه؟ أجب بـ نعم أو لا. قدم التصنيف فقط دون شرح.'
    return [
        {
            "role": "system",
            "content": "أنت خبير في تحليل وتصنيف التغريدات.",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
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
