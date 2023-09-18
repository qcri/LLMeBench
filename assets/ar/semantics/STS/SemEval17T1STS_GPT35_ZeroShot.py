import re

from llmebench.datasets import SemEval17T1STSDataset
from llmebench.models import LegacyOpenAIModel
from llmebench.tasks import STSTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-35-turbo (version 0301)",
        "description": "GPT35 model hosted on Azure, using the Completion API. API version '2023-03-15-preview'.",
        "scores": {"PC": "0.799"},
    }


def config():
    return {
        "dataset": SemEval17T1STSDataset,
        "task": STSTask,
        "model": LegacyOpenAIModel,
        "model_args": {
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": f"Given two sentences, produce a continuous valued similarity score on a scale from 0 to 5, with 0 indicating that the semantics of the sentences are completely independent and 5 signifying semantic equivalence. The output should be exactly in form Similarity score =.\n{input_sample}",
            }
        ],
    }


def post_process(response):
    raw_response = response["choices"][0]["text"]

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
