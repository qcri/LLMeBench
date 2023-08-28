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
            "api_url": os.environ["API_URL"],
            "class_labels": ["TRUE", "FALSE"],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/sarcasm/ArSarcasm-2/testing_data.csv"
        },
    }


def prompt(input_sample):
    prompt_string = (
            'Predict whether the following "tweet" is sarcastic. Return "yes" if the tweet is sarcastic and "no" if the tweet is not sarcastic. Provide only label.\n\ntweet: '
            + input_sample
            + "\n"
              "label: \n"
    )
    return {
        "prompt": prompt_string,
    }


def post_process(response):
    label = response["outputs"].strip().lower()
    label = label.replace("<s>", "").replace("</s>", "")

    if label == "yes":
        return "TRUE"
    elif label == "no":
        return "FALSE"
    else:
        return None