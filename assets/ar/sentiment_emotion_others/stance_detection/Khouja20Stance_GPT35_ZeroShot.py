import os

from llmebench.datasets import Khouja20StanceDataset
from llmebench.models import GPTModel, RandomGPTModel
from llmebench.tasks import Khouja20StanceTask


def config():
    return {
        "dataset": Khouja20StanceDataset,
        "dataset_args": {},
        "task": Khouja20StanceTask,
        "task_args": {},
        "model": GPTModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": "gpt",
            "class_labels": ["agree", "disagree"],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/factuality_stance_khouja/stance/test.csv"
        },
    }


def prompt(input_sample):
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": f'Can you check if first sentence agree or disagree with second sentence? Say only agree or disagree.\n\n first-sentence: {input_sample["sentence_1"]}\nsecond-sentence: {input_sample["sentence_2"]}\n label: \n',
            }
        ],
    }


def post_process(response):
    label = response["choices"][0]["text"].lower().replace(".", "")

    return label
