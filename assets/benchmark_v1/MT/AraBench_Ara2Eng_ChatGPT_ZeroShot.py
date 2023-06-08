import os

from arabic_llm_benchmark.datasets import AraBenchDataset
from arabic_llm_benchmark.models import GPTModel, RandomGPTModel
from arabic_llm_benchmark.tasks import MachineTranslationTask


def config():
    return {
        "dataset": AraBenchDataset,
        "dataset_args": {
            "src": "madar.test.mgr.0.dz.ar",
            "tgt": "madar.test.mgr.0.dz.en",
        },
        "task": MachineTranslationTask,
        "task_args": {},
        "model": GPTModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": "gpt",
            "max_tries": 3,
        },
        "general_args": {"data_path": "data/MT/"},
    }


def prompt(input_sample):
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": f"Translate the following to English, output only the translation:\n {input_sample}",
            }
        ],
    }


def post_process(response):
    return response["choices"][0]["text"]
