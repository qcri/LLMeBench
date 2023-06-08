import os
import re

from arabic_llm_benchmark.datasets import STSTrack1Dataset
from arabic_llm_benchmark.models import GPTModel, RandomGPTModel
from arabic_llm_benchmark.tasks import STSTrack1Task


def config():
    return {
        "dataset": STSTrack1Dataset,
        "dataset_args": {},
        "task": STSTrack1Task,
        "task_args": {},
        "model": GPTModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": "gpt",
            "class_labels": "NA",
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/STS/semeval-2017/STS2017.eval.v1.1/STS.input.track1.ar-ar.txt;data/STS/semeval-2017/STS2017.gs/STS.gs.track1.ar-ar.txt",
            # "ground_truth_data_path": "data/STS/semeval-2017/STS2017.eval.v1.1/STS.input.track1.ar-ar.txt"
        },
    }


def prompt(input_sample):
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": f"Given two sentences, produce a continuous valued similarity score on a scale from 0 to 5, with 0  indicating that the semantics of the sentences are completely independent and 5 signifying semantic equivalence. The output should be exactly in form Similarity score =.\n {input_sample}",
            }
        ],
    }


def post_process(response):
    raw_response = response["choices"][0]["text"]
    regex_float = r"\b\d+\.\d+\.?\b"

    try:
        if "Similarity score =" in raw_response:
            # output = raw_response.replace("Similarity score = ", "")
            match = re.findall(regex_float, raw_response)[0]
            score = float(match)

    except Exception as e:
        score = None

    return score
