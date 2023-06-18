import os

from arabic_llm_benchmark.datasets import ArabicParsingDataset
from arabic_llm_benchmark.models import GPTModel, RandomGPTModel
from arabic_llm_benchmark.tasks import ArabicParsingTask


def config():
    return {
        "dataset": ArabicParsingDataset,
        "dataset_args": {},
        "task": ArabicParsingTask,
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
            "data_path": "data/sequence_tagging_ner_pos_etc/Parsing/arabic_PADT_test_blind.conll"
        },
    }


def prompt(input_sample):
    return {
        "system_message": "You are a linguist that helps in annotating data.",
        "messages": [
            {
                "sender": "user",
                "text": f"Given the following features (in order: ID, Form, Lemma, CPostTag, POSTag, Features), predict the Head of each token in the following sentence, which is either a value of a related ID or 0. A value of zero means the token attaches to the virtual root node: {input_sample}"
            }
        ],
    }


def post_process(response):
    return response["choices"][0]["text"]
