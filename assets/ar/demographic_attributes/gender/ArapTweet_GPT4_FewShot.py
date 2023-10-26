from llmebench.datasets import ArapTweetDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import ClassificationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. 3 samples where chosen per test sample based on MaxMarginalRelevance for few shot learning.",
        "scores": {"Macro-F1": "0.980"},
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


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n"
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
    else:
        fixed_label = None

    return fixed_label
