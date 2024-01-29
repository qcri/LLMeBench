from llmebench.datasets import ArabGendDataset
from llmebench.models import FastChatModel
from llmebench.tasks import ClassificationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Jais-13b",
        "description": "Locally hosted Jais-13b-chat model using FastChat.",
        "scores": {"Macro-F1": "0.674"},
    }


def config():
    return {
        "dataset": ArabGendDataset,
        "task": ClassificationTask,
        "model": FastChatModel,
        "model_args": {
            "class_labels": ["m", "f"],
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    base_prompt = (
        f"Identify the gender from the following name as 'female' or 'male'.\n\n"
        f"name: {input_sample}"
        f"gender: \n"
    )
    return [
        {
            "role": "user",
            "content": base_prompt,
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"]
    if label.lower() == "male":
        return "m"
    elif "female" in label.lower():
        return "f"
    else:
        return None
