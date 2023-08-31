import os

from llmebench.datasets import ArapTweetDataset
from llmebench.models import GPTChatCompletionModel
from llmebench.tasks import DemographyGenderTask


def config():
    return {
        "dataset": ArapTweetDataset,
        "dataset_args": {},
        "task": DemographyGenderTask,
        "task_args": {},
        "model": GPTChatCompletionModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
            "class_labels": ["Female", "Male"],
            "max_tries": 30,
        },
        "general_args": {
            "data_path": "data/demographic_attributes/gender/test-ARAP-unique.txt"
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
