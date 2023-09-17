from llmebench.datasets import OSACT4SubtaskADataset
from llmebench.models import PetalsModel
from llmebench.tasks import OffensiveTask


def config():
    return {
        "dataset": OSACT4SubtaskADataset,
        "dataset_args": {},
        "task": OffensiveTask,
        "task_args": {},
        "model": PetalsModel,
        "model_args": {
            "class_labels": ["OFF", "NOT_OFF"],
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    return {
        "prompt": 'Given the following Arabic tweet, label it as "OFF" if it contains offensive content, or label it as "NOT_OFF" otherwise, based on the content of the tweet. Provide only label.\n\n'
        + "sentence: "
        + input_sample
        + "label: "
    }


def post_process(response):
    label = response["outputs"].strip()
    label = label.replace("<s>", "")
    label = label.replace("</s>", "")

    return label
