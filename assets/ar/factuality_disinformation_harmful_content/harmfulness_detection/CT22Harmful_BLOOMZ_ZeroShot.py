import os

from llmebench.datasets import CT22HarmfulDataset
from llmebench.models import PetalsModel
from llmebench.tasks import HarmfulDetectionTask


def config():
    return {
        "dataset": CT22HarmfulDataset,
        "dataset_args": {},
        "task": HarmfulDetectionTask,
        "task_args": {},
        "model": PetalsModel,
        "model_args": {
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/harmful/CT22_arabic_1C_harmful_test_gold.tsv"
        },
    }


def prompt(input_sample):
    prompt = (
        "Classify the sentence by whether the information in it can cause harm to the society, an entity or persona. Answer only by yes or no. Provide only label.\n\n"
        + "Sentence: "
        + input_sample
        + "\nLabel: \n"
    )

    return {
        "prompt": prompt,
    }


def post_process(response):
    input_label = (
        response["outputs"].strip().lower().replace("<s>", "").replace("</s>", "")
    )
    pred_label = ""

    if (
        input_label.startswith("harmful")
        or input_label.startswith("yes")
        or "label: 1" in input_label
        or "label: yes" in input_label
        or "label: harmful" in input_label
    ):
        pred_label = "1"

    if (
        (
            input_label.startswith("no")
            or input_label == "label: safe"
            or "not harmful" in input_label
        )
        or "label: 0" in input_label
        or "label: no" in input_label
        or "label: not harmful" in input_label
    ):
        pred_label = "0"

    if pred_label == "":
        pred_label = None

    return pred_label
