import os

from arabic_llm_benchmark.datasets import ArabicPOSDataset
from arabic_llm_benchmark.models import GPTModel, RandomGPTModel
from arabic_llm_benchmark.tasks import ArabicPOSTask


def config():
    return {
        "dataset": ArabicPOSDataset,
        "dataset_args": {},
        "task": ArabicPOSTask,
        "task_args": {},
        "model": GPTModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": "gpt",
            # "class_labels": ["m", "f"],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/sequence_tagging_ner_pos_etc/POS/egy.pos/egy.data_5.test.src-trg.sent"
        },
    }


def prompt(input_sample):
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": f"Assign POS tag to each morphological segment within each word. group the tags for each word with +: {input_sample}"
                + ".\nThe output should be in the format: [{word: label}, {word: label}]",
            }
        ],
    }


def post_process(response):
    return response["choices"][0]["text"]
