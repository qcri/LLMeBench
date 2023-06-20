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
            "data_path": {
                "sentences_path": "data/STS/semeval-2017/STS2017.eval.v1.1/STS.input.track2.ar-en.txt",
                "gt_data_path": "data/STS/semeval-2017/STS2017.gs/STS.gs.track2.ar-en.txt",
            }
        }
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

    try:
        return float(pred_num) / 2
    except:
        print("Label issue! " + str(label))
        return None