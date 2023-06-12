import os

from arabic_llm_benchmark.datasets import ArabicSegmentationDataset
from arabic_llm_benchmark.models import GPTModel, RandomGPTModel
from arabic_llm_benchmark.tasks import ArabicSegmentationTask


def config():
    return {
        "dataset": ArabicSegmentationDataset,
        "dataset_args": {},
        "task": ArabicSegmentationTask,
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
            "data_path": "data/sequence_tagging_ner_pos_etc/segmentation/egy.seg/egy.data_5.test.src.sent"
        },
    }


def prompt(input_sample):
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": f"A word can be composed of one root and one or multiple affixed, segment the following sentence into its morphological constituents: {input_sample}"
                + ".\nThe output should be in the format: [{word: {prefix}(0,n)+root+{suffix}(0,n)},...]",
            }
        ],
    }


def post_process(response):
    return response["choices"][0]["text"]
