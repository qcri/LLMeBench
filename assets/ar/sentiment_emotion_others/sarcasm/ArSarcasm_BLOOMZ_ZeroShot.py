from llmebench.datasets import ArSarcasmDataset

from llmebench.models import PetalsModel

from llmebench.tasks import SarcasmTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "bloomz-176b (8bit quantized)",
        "description": "Locally hosted BLOOMZ 176b model (8 bit quantized version) using the Petals.",
        "scores": {"F1 (POS)": "0.286"},
    }


def config():
    return {
        "dataset": ArSarcasmDataset,
        "task": SarcasmTask,
        "model": PetalsModel,
        "model_args": {
            "class_labels": ["TRUE", "FALSE"],
            "max_tries": 3,
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
