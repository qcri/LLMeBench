import os

from arabic_llm_benchmark.datasets import NameInfoDataset
from arabic_llm_benchmark.models import GPTModel, RandomGPTModel
from arabic_llm_benchmark.tasks import DemographyNameInfoTask


def config():
    return {
        "dataset": NameInfoDataset,
        "dataset_args": {},
        "task": DemographyNameInfoTask,
        "task_args": {},
        "model": GPTModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
            "class_labels": ["m", "f"],
            "max_tries": 3,
        },
        "general_args": {"data_path": "data/demography/name_info/wikidata_test.txt"},
    }


def prompt(input_sample):
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": f"Predict the country of citizenship of the following person name. Write ONLY the country code in ISO 3166-1 alpha-2 format without explananation.\n {input_sample}",
            }
        ],
    }


def post_process(response):
    out = response["choices"][0]["text"]
    return out.lower()
