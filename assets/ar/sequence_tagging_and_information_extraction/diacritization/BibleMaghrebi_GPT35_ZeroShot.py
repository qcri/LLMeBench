import os

from llmebench.datasets import BibleMaghrebiDiacritizationDataset
from llmebench.models import GPTModel
from llmebench.tasks import ArabicDiacritizationTask


def config():
    sets = [
        ("mor", "morrocan_f05.test.src-tgt.txt"),
        ("tun", "tunisian_f05.test.src-tgt.txt"),
    ]
    configs = []
    for name, testset in sets:
        configs.append(
            {
                "name": name,
                "config": {
                    "dataset": BibleMaghrebiDiacritizationDataset,
                    "dataset_args": {},
                    "task": ArabicDiacritizationTask,
                    "task_args": {},
                    "model": GPTModel,
                    "model_args": {
                        "api_type": "azure",
                        "api_version": "2023-03-15-preview",
                        "api_base": os.environ["AZURE_API_URL"],
                        "api_key": os.environ["AZURE_API_KEY"],
                        "engine_name": os.environ["ENGINE_NAME"],
                        "max_tries": 3,
                    },
                    "general_args": {
                        "data_path": f"data/sequence_tagging_ner_pos_etc/diacritization/{testset}"
                    },
                },
            }
        )

    return configs


def prompt(input_sample):
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": f"Diacritize fully the following Arabic sentence: {input_sample}",
            }
        ],
    }


def post_process(response):
    return response["choices"][0]["text"]
