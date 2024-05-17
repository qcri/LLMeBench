from llmebench.datasets import HuggingFaceDataset
from llmebench.models import FastChatModel
from llmebench.tasks import ClassificationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Llama-2-13b-chat-hf",
        "description": "Locally hosted Llama-2-13b-chat hf model using FastChat.",
        "scores": {"Accuracy": "0.924"},
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
        "model": FastChatModel,
        "general_args": {"custom_test_split": "validation"},
    }


def prompt(input_sample):
    prompt_string = (
        f"You are tasked with analyzing the sentiment of the given sentence. "
        f"Please read it carefully and determine whether the sentiment expressed is positive or negative. Provide only label.\n\n"
        f"sentence: {input_sample.strip()}\n"
        f"label:\n"
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a data annotation expert specializing in sentiment analysis."
            ),
        },
        {"role": "user", "content": prompt_string},
    ]


def post_process(response):
    mapping = {"positive": 1, "negative": 0}

    pred_label = response["choices"][0]["message"]["content"].lower()

    if "\n\nlabel: negative" in pred_label:
        pred_label = "negative"
    elif "\n\nlabel: positive" in pred_label:
        pred_label = "positive"
    elif "\n\nlabel:" in pred_label:
        pred_label = pred_label.split("\n\nlabel:")[1]
        pred_label = pred_label.strip().lower()
    if pred_label == "positive" or pred_label == "negative":
        return mapping[pred_label]
    elif "\n\nnegative" in pred_label:
        pred_label = "negative"
    elif "\n\npositive" in pred_label:
        pred_label = "positive"
    else:
        pred_label = None

    if pred_label is not None:
        return mapping[pred_label]
    else:
        return None
