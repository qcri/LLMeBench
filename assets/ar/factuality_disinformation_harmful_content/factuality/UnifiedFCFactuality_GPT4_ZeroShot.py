from llmebench.datasets import UnifiedFCFactualityDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import FactualityTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'.",
        "scores": {"Macro-F1": "0.581"},
    }


def config():
    return {
        "dataset": UnifiedFCFactualityDataset,
        "task": FactualityTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["true", "false"],
            "max_tries": 30,
        },
    }


def prompt(input_sample):
    prompt_string = (
        f'Annotate the "text" into one of the following categories: correct or incorrect\n\n'
        f"text: {input_sample}\n"
        f"label: \n"
    )
    return [
        {
            "role": "system",
            "content": "You are a news analyst and you can check the information in the news article and annotate them.",  # You are capable of identifying and annotating tweets correct or incorrect
        },
        {
            "role": "user",
            "content": prompt_string,
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"].lower()
    # label_fixed = label.replace("label:", "").strip()

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
    else:
        label_fixed = None

    return label_fixed
