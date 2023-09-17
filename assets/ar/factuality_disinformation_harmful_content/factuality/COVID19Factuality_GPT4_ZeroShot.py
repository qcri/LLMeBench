from llmebench.datasets import COVID19FactualityDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import FactualityTask


def config():
    return {
        "dataset": COVID19FactualityDataset,
        "dataset_args": {},
        "task": FactualityTask,
        "task_args": {},
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["yes", "no"],
            "max_tries": 30,
        },
    }


def prompt(input_sample):
    prompt_string = (
        f'Annotate the "tweet" into one of the following categories: correct or incorrect\n\n'
        f"tweet: {input_sample}\n"
        f"label: \n"
    )
    return [
        {
            "role": "system",
            "content": "You are a social media expert, a fact-checker and you can annotate tweets.",  # You are capable of identifying and annotating tweets correct or incorrect
        },
        {
            "role": "user",
            "content": prompt_string,
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"]

    if label.startswith("I am unable to verify".lower()) or label.startswith(
        "I am unable to categorize".lower()
    ):
        label_fixed = None
    elif "label: incorrect" in label or "incorrect" in label:
        label_fixed = "no"
    elif "label: correct" in label or "correct" in label:
        label_fixed = "yes"
    else:
        label_fixed = None

    return label_fixed
