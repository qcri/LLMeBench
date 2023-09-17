from llmebench.datasets import SpamDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import SpamTask


def config():
    return {
        "dataset": SpamDataset,
        "dataset_args": {},
        "task": SpamTask,
        "task_args": {},
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["__label__ADS", "__label__NOTADS"],
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    return [
        {
            "role": "system",
            "content": "You are an AI assistant that helps people find information.",
        },
        {
            "role": "user",
            "content": f"If the following sentence can be classified as spam or contains an advertisemnt, write '__label__ADS' without explnanation, otherwise write '__label__NOTADS' without explanantion.\n {input_sample}\n",
        },
    ]


def post_process(response):
    out = response["choices"][0]["message"]["content"]
    j = out.find(".")
    if j > 0:
        out = out[0:j]
    return out
