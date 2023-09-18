from llmebench.datasets import ArSarcasm2Dataset
from llmebench.models import LegacyOpenAIModel
from llmebench.tasks import SarcasmTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-35-turbo (version 0301)",
        "description": "GPT35 model hosted on Azure, using the Completion API. API version '2023-03-15-preview'.",
        "scores": {"F1 (POS)": "0.537"},
    }


def config():
    return {
        "dataset": ArSarcasm2Dataset,
        "dataset_args": {},
        "task": SarcasmTask,
        "task_args": {},
        "model": LegacyOpenAIModel,
        "model_args": {
            "class_labels": ["TRUE", "FALSE"],
            "max_tries": 1,
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
    label = response["choices"][0]["text"].strip().lower()
    if "yes" in label:
        return "TRUE"
    elif "no" in label:
        return "FALSE"

    return None
