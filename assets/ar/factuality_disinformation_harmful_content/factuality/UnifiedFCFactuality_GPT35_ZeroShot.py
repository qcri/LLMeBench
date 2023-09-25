from llmebench.datasets import UnifiedFCFactualityDataset
from llmebench.models import LegacyOpenAIModel
from llmebench.tasks import FactualityTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-35-turbo (version 0301)",
        "description": "GPT35 model hosted on Azure, using the Completion API. API version '2023-03-15-preview'.",
        "scores": {"Macro-F1": "0.306"},
    }


def config():
    return {
        "dataset": UnifiedFCFactualityDataset,
        "task": FactualityTask,
        "model": LegacyOpenAIModel,
        "model_args": {
            "class_labels": ["true", "false"],
            "max_tries": 30,
        },
    }


def prompt(input_sample):
    prompt_string = (
        f"Classify the text as only true or false. Provide only label.\n\n"
        f"text: {input_sample}\n"
        f"label: \n"
    )
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": prompt_string,
            }
        ],
    }


def post_process(response):
    label = response["choices"][0]["text"].lower().replace(".", "").lower()

    if (
        label.startswith("I am unable to verify".lower())
        or label.startswith("I am unable to categorize".lower())
        or label.startswith(
            "I am an AI language model and I am unable to verify".lower()
        )
    ):
        label_fixed = None
    elif "label: incorrect" in label or "incorrect" in label:
        label_fixed = "false"
    elif "label: correct" in label or "correct" in label:
        label_fixed = "true"
    elif "true" == label or "false" == label:
        label_fixed = label
    else:
        label_fixed = None

    return label_fixed
