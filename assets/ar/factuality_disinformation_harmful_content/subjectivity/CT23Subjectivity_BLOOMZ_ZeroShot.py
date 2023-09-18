from llmebench.datasets import CT23SubjectivityDataset
from llmebench.models import PetalsModel
from llmebench.tasks import SubjectivityTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "bloomz-176b (8bit quantized)",
        "description": "Locally hosted BLOOMZ 176b model (8 bit quantized version) using the Petals.",
        "scores": {"Macro-F1": "0.428"},
    }


def config():
    return {
        "dataset": CT23SubjectivityDataset,
        "task": SubjectivityTask,
        "model": PetalsModel,
        "model_args": {
            "class_labels": ["SUBJ", "OBJ"],
            "max_tries": 3,
        },
        "general_args": {"test_split": "ar/dev"},
    }


def prompt(input_sample):
    return {
        "prompt": f"Classify the sentence as Subjective or Objective. Provide only label.\n\ntext: {input_sample}\nlabel: ",
    }


def post_process(response):
    label = response["outputs"].strip().lower()
    label = label.replace("<s>", "")
    label = label.replace("</s>", "")

    if label == "objective" or label == "objective.":
        label_fixed = "OBJ"
    elif label == "subjective" or label == "subjective.":
        label_fixed = "SUBJ"

    return label_fixed
