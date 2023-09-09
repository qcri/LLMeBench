import os

from llmebench.datasets import SemEval17T2STSDataset
from llmebench.models import LegacyOpenAIModel
from llmebench.tasks import STSTask


def config():
    return {
        "dataset": SemEval17T2STSDataset,
        "dataset_args": {},
        "task": STSTask,
        "task_args": {},
        "model": LegacyOpenAIModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
            "class_labels": "NA",
            "max_tries": 3,
        },
        "general_args": {
            "data_path": {
                "sentences_path": "data/STS/semeval-2017/STS2017.eval.v1.1/STS.input.track2.ar-en.txt",
                "gt_data_path": "data/STS/semeval-2017/STS2017.gs/STS.gs.track2.ar-en.txt",
            }
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
