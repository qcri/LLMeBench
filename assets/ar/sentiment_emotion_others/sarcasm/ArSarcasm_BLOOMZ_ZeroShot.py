import os

from llmebench.datasets import ArSarcasmDataset

from llmebench.models import PetalsModel

from llmebench.tasks import SarcasmTask


def config():
    return {
        "dataset": ArSarcasmDataset,
        "dataset_args": {},
        "task": SarcasmTask,
        "task_args": {},
        "model": PetalsModel,
        "model_args": {
            "api_url": os.environ["API_URL"],
            "class_labels": ["TRUE", "FALSE"],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/sentiment_emotion_others/sarcasm/ArSarcasm/ArSarcasm_test.csv"
        },
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
