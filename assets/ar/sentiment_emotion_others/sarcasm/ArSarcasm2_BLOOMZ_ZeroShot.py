from llmebench.datasets import ArSarcasm2Dataset
from llmebench.models import PetalsModel
from llmebench.tasks import SarcasmTask


def config():
    return {
        "dataset": ArSarcasm2Dataset,
        "dataset_args": {},
        "task": SarcasmTask,
        "task_args": {},
        "model": PetalsModel,
        "model_args": {
            "class_labels": ["TRUE", "FALSE"],
            "max_tries": 3,
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
