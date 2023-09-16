from llmebench.datasets import CT22CheckworthinessDataset
from llmebench.models import LegacyOpenAIModel
from llmebench.tasks import CheckworthinessTask


def config():
    return {
        "dataset": CT22CheckworthinessDataset,
        "dataset_args": {},
        "task": CheckworthinessTask,
        "task_args": {},
        "model": LegacyOpenAIModel,
        "model_args": {
            "class_labels": ["0", "1"],
            "max_tries": 3,
        },
        "general_args": {"test_split": "ar"},
    }


def prompt(input_sample):
    ## GPT 3.5 - turbo

    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": f'Classify the "tweet" as checkworthy or not_checkworthy. Provide only label.\n\nsentence: {input_sample}label: ',
            }
        ],
    }


def post_process(response):
    label = response["choices"][0]["text"]

    if label == "checkworthy":
        label_fixed = "1"
    elif label == "Not_checkworthy." or label == "not_checkworthy":
        label_fixed = "0"

    return label_fixed
