import os

from arabic_llm_benchmark.datasets import Khouja20ClaimDataset
from arabic_llm_benchmark.models import GPTModel, RandomGPTModel
from arabic_llm_benchmark.tasks import FactClaimTask


def config():
    return {
        "dataset": Khouja20ClaimDataset,
        "dataset_args": {},
        "task": FactClaimTask,
        "task_args": {},
        "model": GPTModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": "gpt",
            "class_labels": ["0", "1"],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/factuality_stance_khouja/claim/test.csv"
        },
    }


def prompt(input_sample):
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": f"Does this sentence contain a factual claim? Answer only by yes or no.\n {input_sample}",
            }
        ],
    }


def post_process(response):
    raw_response = response["choices"][0]["text"].lower().replace(".", "")

    mapping = {
        "no": "0",
        "yes": "1"
    }

    return mapping[raw_response]
