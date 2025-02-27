from llmebench.datasets import ArapTweetDataset
from llmebench.models import FastChatModel
from llmebench.tasks import ClassificationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Jais-13b",
        "description": "Locally hosted Jais-13b-chat model using FastChat.",
        "scores": {"Macro-F1": ""},
    }


def config():
    return {
        "dataset": ArapTweetDataset,
        "task": ClassificationTask,
        "model": FastChatModel,
        "model_args": {
            "class_labels": ["Female", "Male"],
            "max_tries": 30,
        },
    }


def prompt(input_sample):
    base_prompt = (
        f"Identify the gender from the following name as 'Female' or 'Male'.\n\n"
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
    # label = label.replace("gender:", "").strip()
    if "gender: Female" in label or "\nFemale" in label or label == "Female":
        label = "Female"
    elif (
        "gender: Male" in label
        or "\nMale" in label
        or "likely to be 'Male'" in label
        or label == "Male"
        or "typically a 'Male' name" in label
    ):
        label = "Male"
    else:
        label = None

    return label
