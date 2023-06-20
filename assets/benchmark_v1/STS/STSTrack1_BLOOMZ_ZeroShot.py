import os
import re

from arabic_llm_benchmark.datasets import STSArSemEval17Track1Dataset
from arabic_llm_benchmark.models import BLOOMPetalModel
from arabic_llm_benchmark.tasks import STSTrack1Task


def config():
    return {
        "dataset": STSArSemEval17Track1Dataset,
        "dataset_args": {},
        "task": STSTrack1Task,
        "task_args": {},
        "model": BLOOMPetalModel,
        "model_args": {
            "api_url": os.environ["API_URL"],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/STS/semeval-2017",
        },
    }


def prompt(input_sample):
    s1,s2 = input_sample.split("\t")

    prompt_string = (
        f"Given two sentences, produce similarity score on a scale from 0 to 5, with 0 indicating that the semantics of the sentences are completely independent and 5 signifying semantic equivalence. "
        f"\n\nsentence1: {s1}\nSentence2: {s2}\nThe output should be exactly in form score=\n"
    )
    return {
            "prompt": prompt_string
        }


def post_process(response):
    label = response["outputs"].strip().lower()
    label = label.replace("<s>", "")
    label = label.replace("</s>", "")

    if "score=" in label:
        pred_num = (
            label.split("score= ")[1]
            .strip()
            .split(" ")[0]
            .rstrip(".")
        )
        score = float(pred_num)
    else:
        print("Issue with label!" + label)
        score = None

    return score
