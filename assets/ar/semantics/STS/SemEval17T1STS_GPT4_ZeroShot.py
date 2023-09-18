import re

from llmebench.datasets import SemEval17T1STSDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import STSTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'.",
        "scores": {"PC": "0.813"},
    }


def config():
    return {
        "dataset": SemEval17T1STSDataset,
        "dataset_args": {},
        "task": STSTask,
        "task_args": {},
        "model": OpenAIModel,
        "model_args": {
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    prompt_string = (
        f"Given two sentences, produce a continuous valued similarity score on a "
        f"scale from 0 to 5, with 0 indicating that the semantics of the sentences are "
        f"completely independent and 5 signifying semantic equivalence. The output "
        f"should be exactly in form Similarity score =. \n{input_sample}"
    )
    return [
        {
            "role": "system",
            "content": "You are an AI assistant that helps people find information.",
        },
        {
            "role": "user",
            "content": prompt_string,
        },
    ]


def post_process(response):
    raw_response = response["choices"][0]["message"]["content"]

    if "Similarity score =" in raw_response:
        pred_num = (
            raw_response.split("Similarity score = ")[1]
            .strip()
            .split(" ")[0]
            .rstrip(".")
        )
        score = float(pred_num)
    else:
        score = None

    return score
