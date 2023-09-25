from llmebench.datasets import ANSStanceDataset
from llmebench.models import PetalsModel
from llmebench.tasks import StanceTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "bloomz-176b (8bit quantized)",
        "description": "Locally hosted BLOOMZ 176b model (8 bit quantized version) using the Petals.",
        "scores": {"Macro-F1": "0.223"},
    }


def config():
    return {
        "dataset": ANSStanceDataset,
        "task": StanceTask,
        "model": PetalsModel,
        "model_args": {
            "class_labels": ["agree", "disagree"],
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    prompt = f'Can you check if first sentence agree or disagree with second sentence? Say only agree or disagree.\n\n first-sentence: {input_sample["sentence_1"]}\nsecond-sentence: {input_sample["sentence_2"]}\n label: \n'

    return {"prompt": prompt}


def post_process(response):
    label = response["outputs"].strip().lower()
    label = label.replace("<s>", "")
    label = label.replace("</s>", "")

    return label
