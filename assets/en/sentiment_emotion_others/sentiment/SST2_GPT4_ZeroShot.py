from llmebench.datasets import SST2
from llmebench.models import OpenAIModel
from llmebench.tasks import ClassificationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. Uses an prompt specified in Arabic.",
        # "scores": {"Macro-F1": "0.547"},
    }


def config():
    return {
        "dataset": SST2,
        "task": ClassificationTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["positive", "negative"],
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    prompt_string = (
        f"You are tasked with analyzing the sentiment of the given sentence. "
        f"Please read it carefully and determine whether the sentiment expressed is positive or negative. Provide only label.\n\n"
        f"sentence: {input_sample}\n"
        f"label:\n"
    )
    return [
        {
            "role": "system",
            "content": "You are a data annotation expert specializing in sentiment analysis.",
        },
        {"role": "user", "content": prompt_string},
    ]


def post_process(response):
    if not response:
        return None
    label = response["choices"][0]["message"]["content"].lower()

    label_fixed = label.replace("label:", "").replace("sentiment: ", "").strip()
    if label_fixed.startswith("Please provide the text"):
        label_fixed = None

    return label_fixed
