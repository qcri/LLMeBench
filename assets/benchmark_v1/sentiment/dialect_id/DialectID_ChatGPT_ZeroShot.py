import os

from arabic_llm_benchmark.datasets import DialectIDDataset
from arabic_llm_benchmark.models import GPTModel, RandomGPTModel
from arabic_llm_benchmark.tasks import DialectIDTask


def config():
    return {
        "dataset": DialectIDDataset,
        "dataset_args": {},
        "task": DialectIDTask,
        "task_args": {},
        "model": GPTModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
            "class_labels": ["__label__EG", "__label__DZ", "__label__SD", "__label__YE", "__label__SY", "__label__AE", "__label__JO", "__label__LY", "__label__PS", "__label__OM", "__label__QA", "__label__BH", "__label__MSA", "__label__SA", "__label__IQ", "__label__MA"],          

            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/sentiment_emotion_others/dialect_id/QADI_test-PalestinePS-corrected.txt"
        },
    }


def prompt(input_sample):
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": f"Write only the country code of the Arabic country in which this sentence is written in its dialect without any explanation. Write only the country code in ISO 3166-1 alpha-2 format without explanation. Write 'MSA' if the sentence is written in Modern Standard Arabic.\n {input_sample}",
            }
        ],
    }


def post_process(response):
    out = response["choices"][0]["text"]
    j = out.find(".")
    if j > 0:
        out = out[0:j]
    return out
