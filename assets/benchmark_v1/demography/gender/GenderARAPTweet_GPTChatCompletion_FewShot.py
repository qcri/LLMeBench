import os

from arabic_llm_benchmark.datasets import ArapTweetDataset
from arabic_llm_benchmark.models import GPTChatCompletionModel
from arabic_llm_benchmark.tasks import DemographyGenderTask


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
            "data_path": "data/demographic_attributes/gender/test-ARAP-unique.txt",
            "fewshot": {
                "train_data_path": "data/demographic_attributes/gender/train-wajdi.tsv",
                "deduplicate": True,
            },
        },
    }


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n\n"
    out_prompt = out_prompt + "Here are some examples:\n\n"

    for index, example in enumerate(examples):
        out_prompt = (
            out_prompt
            + "Example "
            + str(index)
            + ":"
            + "\n"
            + "name: "
            + example["input"]
            + "\ngender: "
            + example["label"]
            + "\n\n"
        )

    # Append the sentence we want the model to predict for but leave the Label blank
    out_prompt = out_prompt + "name: " + input_sample + "\ngender: \n"

    return out_prompt


def prompt(input_sample, examples):
    base_prompt = f"Identify the gender from the following name as 'Female' or 'Male'."
    return [
        {
            "role": "system",
            "content": "You are an expert to identify the gender from a person's name.",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"]

    if (
        "Female." in label
        or "\nFemale" in label
        or "gender: Female" in label
        or label == "Female"
    ):
        fixed_label = "Female"
    elif (
        "Male." in label
        or "\nMale" in label
        or "gender: Male" in label
        or label == "Male"
    ):
        fixed_label = "Male"
    elif label.startswith("I'm sorry, but"):
        fixed_label = None
    else:
        fixed_label = None

    return fixed_label
