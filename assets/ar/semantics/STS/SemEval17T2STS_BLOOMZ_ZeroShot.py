import re

from llmebench.datasets import SemEval17T2STSDataset
from llmebench.models import PetalsModel
from llmebench.tasks import STSTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "bloomz-176b (8bit quantized)",
        "description": "Locally hosted BLOOMZ 176b model (8 bit quantized version) using the Petals.",
        "scores": {"PC": "0.512"},
    }


def config():
    return {
        "dataset": SemEval17T2STSDataset,
        "task": STSTask,
        "model": PetalsModel,
        "model_args": {
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    s1, s2 = input_sample.split("\t")

    prompt_string = (
        # f"Given two sentences, produce similarity score from 0 to 5, with 0 indicating that the semantics of the sentences are independent and 5 signifying semantic equivalence. "
        f"Given two sentences, provide a similarity score from 0 to 10, with 10 meaning that the semantics of the sentence are equivalence and 0 meaning that sentences are independent."
        f"\nsentence1: {s1}\nSentence2: {s2}\n similarity score = "
    )
    return {"prompt": prompt_string}


def post_process(response):
    label = response["outputs"].strip().lower()
    label = label.replace("<s>", "")
    label = label.replace("</s>", "")
    pattern = r"\b\d+\.\d*|\d+\b"
    pred_num = re.findall(pattern, label)[0]

    return float(pred_num) / 2
