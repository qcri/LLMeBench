from llmebench.datasets import ArSarcasmDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import SarcasmTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'.",
        "scores": {"F1 (POS)": "0.400"},
    }


def config():
    return {
        "dataset": ArSarcasmDataset,
        "task": SarcasmTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["TRUE", "FALSE"],
            "max_tries": 3,
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
