from llmebench.datasets import CT22CheckworthinessDataset
from llmebench.models import PetalsModel
from llmebench.tasks import CheckworthinessTask


def config():
    return {
        "dataset": CT22CheckworthinessDataset,
        "dataset_args": {},
        "task": CheckworthinessTask,
        "task_args": {},
        "model": PetalsModel,
        "model_args": {
            "class_labels": ["0", "1"],
            "max_tries": 3,
        },
        "general_args": {"test_split": "nl"},
    }


def prompt(input_sample):
    return {
        "prompt": "Classify the tweet as checkworthy or not checkworthy. Provide only label.\n\n"
        + "tweet: "
        + input_sample
        + "label: \n"
    }


def post_process(response):
    label = response["outputs"].strip().lower()
    label = label.replace("<s>", "")
    label = label.replace("</s>", "")

    label_fixed = None

    if label == "checkworthy":
        label_fixed = "1"
    elif (
        label == "Not_checkworthy."
        or label == "not_checkworthy"
        or label == "not checkworthy"
    ):
        label_fixed = "0"

    return label_fixed
