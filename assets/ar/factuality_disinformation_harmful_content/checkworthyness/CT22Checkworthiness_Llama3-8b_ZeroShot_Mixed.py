import re

from llmebench.datasets import CT22CheckworthinessDataset
from llmebench.models import AzureModel
from llmebench.tasks import CheckworthinessTask




def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "Llama-3.1-8B-Instruct",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054)."
    }




def config():
    return {
        "dataset": CT22CheckworthinessDataset,
        "task": CheckworthinessTask,
        "model": AzureModel,
        "model_args": {
            "class_labels": ["0", "1"],
            "max_tries": 30,
        },
        "general_args": {"test_split": "ar"},
    }


def prompt(input_sample):
    base_prompt = (
        ' صنف التغريدة الى "checkworthy" أو "not checkworthy". قم بتقديم التصنيف فقط لأخر تغريدة، لا تقدم أي تبرير إضافي.:\n'
    )
    return [
        {
            "role": "user",
            "content": base_prompt + input_sample,
        },
    ]

import random
def post_process(response):
    print(response)
    if "output" in response:
        # if "content" in response["messages"]:
        label = response["output"].strip()
        label = label.replace("<s>", "")
        label = label.replace("</s>", "")
    else:
        print("Response .. " + str(response))
        label = ""
    label = label.lower()
    
    if "لا أستطيع" in label or "I cannot" in label:
        return random.choice(["0","1"])
    if "لا" in label or "غير" in label or "no" in label or "ليس" in label or "ليست" in label:
        return "0"
    return "1"
    return label
