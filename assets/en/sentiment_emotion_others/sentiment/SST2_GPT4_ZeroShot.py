from llmebench.datasets import HuggingFaceDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import ClassificationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'.",
    }


def config():
    return {
        "dataset": HuggingFaceDataset,
        "dataset_args": {
            "huggingface_dataset_name": "sst2",
            "column_mapping": {
                "input": "sentence",
                "label": "label",
                "input_id": "idx",
            },
        },
        "task": ClassificationTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["positive", "negative"],
            "max_tries": 3,
        },
        "general_args": {"custom_test_split": "validation"},
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

    if label_fixed == "positive":
        return 1
    elif label_fixed == "negative":
        return 0

    return None
