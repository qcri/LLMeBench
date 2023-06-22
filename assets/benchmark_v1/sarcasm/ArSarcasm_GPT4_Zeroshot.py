import os

from arabic_llm_benchmark.datasets import ArSarcasmDataset
from arabic_llm_benchmark.models import GPTChatCompletionModel
from arabic_llm_benchmark.tasks import SarcasmTask


def config():
    return {
        "dataset": ArSarcasmDataset,
        "dataset_args": {},
        "task": SarcasmTask,
        "task_args": {},
        "model": GPTChatCompletionModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
            "class_labels": ["TRUE", "FALSE"],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/sarcasm/ArSarcasm/ArSarcasm_test.csv",
        },
    }


def prompt(input_sample):
    return [
        {
            "role": "system",
            "content": "You are an expert in sarcasm detection.\n\n",
        },
        {
            "role": "user",
            "content": (
                'Predict whether the following "tweet" is sarcastic. Return "yes" if the tweet is sarcastic '
                'and "no" if the tweet is not sarcastic. Provide only label.\n\ntweet: '
                + input_sample
                + "\n"
                "label: \n"
            ),
        },
    ]


def post_process(response):
    content = response["choices"][0]["message"]["content"]
    content = content.strip().lower()
    if "yes" in content:
        return "TRUE"
    elif "no" in content:
        return "FALSE"

    return None
