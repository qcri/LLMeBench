import re

from llmebench.datasets import CT22CheckworthinessDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import CheckworthinessTask




def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "GPT-4o-2024-05-22",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054)."
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
        "general_args": {"test_split": "ar"},
    }


def prompt(input_sample):
    prompt_string = (
        f'هل تحتوي هذه "التغريدة" على ادعاء يستحق التحقق منه؟ أجب بـ checkworthy أو not_checkworthy\n\n'
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

    label = label.replace("label:", "").strip().lower()

    if "لا_يستحق_التحقق" in label or "لا يستحق التحقق" in label or "ليس يستحق التحقق" in label or "لا تستحق التحقق" in label or "no" in label or "لا" in label or "not" in label:
        return "0"
    elif "yes" in label or "نعم" in label or "يستحق التحقق" in label or "checkworthy" in label:
        return "1"
    else:
        return None



