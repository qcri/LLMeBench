import os

from llmebench.datasets import XQuADDataset
from llmebench.models import GPTModel
from llmebench.tasks import QATask


def config():
    return {
        "dataset": XQuADDataset,
        "dataset_args": {},
        "task": QATask,
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
        "general_args": {"data_path": "data/QA/xquad/xquad.ar.json"},
    }


def prompt(input_sample):
    return {
        "system_message": "Assistant is a large language model trained by OpenAI.",
        "messages": [
            {
                "sender": "user",
                "text": f"Your task is to answer questions in Arabic based on a given context.\nNote: Your answers should be spans extracted from the given context without any illustrations.\nYou don't need to provide a complete answer\nContext:{input_sample['context']}\nQuestion:{input_sample['question']}\nAnswer:",
            }
        ],
    }


def post_process(response):
    return response["choices"][0]["text"]