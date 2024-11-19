import re

from llmebench.datasets import CT22CheckworthinessDataset
from llmebench.models import FastChatModel
from llmebench.tasks import CheckworthinessTask





def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "jais-13b-chat",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054)."
    }





def config():
    return {
        "dataset": CT22CheckworthinessDataset,
        "task": CheckworthinessTask,
        "model": FastChatModel,
        "model_args": {
            "class_labels": ["0", "1"],
            "max_tries": 100,
        },
        "general_args": {"test_split": "ar", "fewshot": {"train_split": "ar"}},
    }
def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n"
    out_prompt = out_prompt + "These are some examples:\n\n"
    for index, example in enumerate(examples):
        label = "not checkworthy" if example["label"] == "0" else "checkworthy"

        out_prompt = (
            out_prompt
            + "Example "
            + str(index)
            + ":\n"
            + "Tweet: "
            + example["input"]
            + "\n"
            + "Classification: "
            + label
            + "\n\n"
        )

    out_prompt = out_prompt + "Tweet: " + input_sample + "\nClassification: \n"

    return out_prompt


def prompt(input_sample, examples):
    base_prompt = (
        'Classify the tweet as "checkworthy" or "not checkworthy". Provide the classification only for the last tweet, without providing any additional justification:\n'
    )
    return [
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]

def post_process(response):
    label = response["choices"][0]["message"]["content"]

    label = label.replace("label:", "").strip()
    
  
    label = label.lower()
    if "لا" in label or "not" in label or "no" in label or "ليس" in label or "ليست" in label:
        return "0"
    return "1"
    return label
