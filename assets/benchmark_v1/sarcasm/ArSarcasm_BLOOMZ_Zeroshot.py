import os

from arabic_llm_benchmark.datasets import ArSarcasmDataset

from arabic_llm_benchmark.models import BLOOMPetalModel

from arabic_llm_benchmark.tasks import SarcasmTask


def config():
    return {
        "dataset": ArSarcasmDataset,
        "dataset_args": {},
        "task": SarcasmTask,
        "task_args": {},
        "model": BLOOMPetalModel,
        "model_args": {
            "api_url": os.environ["AZURE_API_URL"],
            "class_labels": ["TRUE", "FALSE"],
            "max_tries": 3,
        },
        "general_args": {"data_path": "data/sarcasm/ArSarcasm/ArSarcasm_testdata.csv"},
    }


def prompt(input_sample):
    prompt_string = (
        f"Predict whether the tweet is sarcastic or not. If it is sarcastic, respond with 'TRUE'. If it is not sarcastic, respond with 'FALSE'.\n\n"
        f"text: {input_sample}\n"
        f"label: "
    )
    return {
        "prompt": prompt_string,
    }


def post_process(response):
    label = response["outputs"].strip().lower()
    label = label.replace("<s>", "").replace("</s>", "")

    if label == "true":
        return "TRUE"
    elif label == "false":
        return "FALSE"
    else:
        return None
