from llmebench.datasets import ArapTweetDataset
from llmebench.models import PetalsModel
from llmebench.tasks import ClassificationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "bloomz-176b (8bit quantized)",
        "description": "Locally hosted BLOOMZ 176b model (8 bit quantized version) using the Petals.",
        "scores": {"Macro-F1": "0.532"},
    }


def config():
    return {
        "dataset": ArapTweetDataset,
        "task": ClassificationTask,
        "model": PetalsModel,
        "model_args": {
            "class_labels": ["Female", "Male"],
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    prompt_string = (
        f"You are an expert to identify the gender from a person's name.\n\n"
        f"Identify the gender from the following name as 'Female' or 'Male'.\n"
        f"Provide only label.\n\n"
        f"name: {input_sample}\n"
        f"label: \n"
    )

    return {
        "prompt": prompt_string,
    }


def post_process(response):
    label = response["outputs"].strip()
    label = label.replace("<s>", "")
    label = label.replace("</s>", "")
    label = label.lower()

    if (
        "female" in label
        or "female." in label
        or "\nfemale" in label
        or label == "female"
    ):
        label = "Female"
    elif "male" in label or "male." in label or "\nmale" in label or label == "male":
        label = "Male"
    else:
        label = None

    return label
