from llmebench.datasets import ArapTweetDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import ClassificationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'.",
        "scores": {"Macro-F1": "0.868"},
    }


def config():
    return {
        "dataset": ArapTweetDataset,
        "task": ClassificationTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["Female", "Male"],
            "max_tries": 30,
        },
    }


def prompt(input_sample):
    prompt_string = (
        f"Identify the gender from the following name as 'Female' or 'Male'.\n\n"
        f"name: {input_sample}"
        f"gender: \n"
    )

    return [
        {
            "role": "system",
            "content": "You are an expert to identify the gender from a person's name.",
        },
        {
            "role": "user",
            "content": prompt_string,
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
