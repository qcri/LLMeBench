import os

from llmebench.datasets import ArSarcasmDataset
from llmebench.models import GPTModel
from llmebench.tasks import SarcasmTask


def config():
    return {
        "dataset": ArSarcasmDataset,
        "dataset_args": {},
        "task": SarcasmTask,
        "task_args": {},
        "model": GPTModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
            "class_labels": ["TRUE", "FALSE"],
            "max_tries": 30,
        },
        "general_args": {
            "data_path": "data/sentiment_emotion_others/sarcasm/ArSarcasm/ArSarcasm_test.csv"
        },
    }


def prompt(input_sample):
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": 'Predict whether the following "tweet" is sarcastic. Return "yes" if the tweet is sarcastic '
                'and "no" if the tweet is not sarcastic. Provide only label.\n\ntweet: '
                + input_sample
                + "\n"
                "label: \n",
            }
        ],
    }


def post_process(response):
    if not response:
        return None

    label = response["choices"][0]["text"]
    content = label.strip().lower()
    if (
        "the tweet is not sarcastic" in content
        or content == "not sarcastic"
        or content == "no"
    ):
        return "FALSE"
    elif (
        content == "yes"
        or "the tweet is sarcastic" in content
        or content == "sarcastic"
    ):
        return "TRUE"
    else:
        return None

    return None
