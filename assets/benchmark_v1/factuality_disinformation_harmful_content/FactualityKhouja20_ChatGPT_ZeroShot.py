import os

from arabic_llm_benchmark.datasets import FactualityKhouja20Dataset
from arabic_llm_benchmark.models import GPTModel, RandomGPTModel
from arabic_llm_benchmark.tasks import FactualityKhouja20Task


def config():
    return {
        "dataset": FactualityKhouja20Dataset,
        "dataset_args": {},
        "task": FactualityKhouja20Task,
        "task_args": {},
        "model": GPTModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": "gpt",
            "class_labels": ["true", "false"],
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
                "text": f"Detect the information in the sentence as correct or incorrect. Use label as true or false.\n\ntext: {input_sample} \nlabel: \n",
            }
        ],
    }


def post_process(response):
    raw_response = response["choices"][0]["text"].lower().replace(".", "")
    return raw_response
