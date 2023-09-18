from llmebench.datasets import AdultDataset
from llmebench.models import PetalsModel
from llmebench.tasks import AdultTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "bloomz-176b (8bit quantized)",
        "description": "Locally hosted BLOOMZ 176b model (8 bit quantized version) using the Petals.",
        "scores": {"Macro-F1": "0.513"},
    }


def config():
    return {
        "dataset": AdultDataset,
        "task": AdultTask,
        "model": PetalsModel,
        "model_args": {
            "class_labels": ["ADULT", "NOT_ADULT"],
            "max_tries": 10,
        },
    }


def prompt(input_sample):
    prompt_string = (
        f"You are an expert annotator, you can identify and label adult content within a tweet.\n"
        f"Given the following tweet, label it as 'ADULT' or 'NOT_ADULT' based on the content of the tweet\n"
        f"Provide only label.\n\n"
        f"tweet: {input_sample}\n"
        f"label: \n"
    )

    return {
        "prompt": prompt_string,
    }


def post_process(response):
    label = response["outputs"].strip()
    label = label.replace("<s>", "")
    label = label.replace("</s>", "")

    label_list = config()["model_args"]["class_labels"]

    if label in label_list:
        label_fixed = label
    else:
        label_fixed = None

    return label_fixed
