import os

from llmebench.datasets import SubjectivityDataset
from llmebench.models import GPTChatCompletionModel
from llmebench.tasks import SubjectivityTask


def config():
    return {
        "dataset": SubjectivityDataset,
        "dataset_args": {},
        "task": SubjectivityTask,
        "task_args": {},
        "model": GPTChatCompletionModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
            "class_labels": ["SUBJ", "OBJ"],
            "max_tries": 30,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/subjectivity/dev_ar.tsv",
            "fewshot": {
                "train_data_path": "data/factuality_disinformation_harmful_content/subjectivity/train_ar.tsv"
            },
        },
    }


def prompt(input_sample, examples):
    base_prompt = (
        "Classify the 'sentence' as subjective or objective. Provide only label.\n"
    )
    return [
        {
            "role": "system",
            "content": "You are social media expert. You can annotate important tweets and require attention from journalists, fact-checker, and government entities.",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n"
    out_prompt = out_prompt + "Here are some examples:\n\n"
    for index, example in enumerate(examples):
        label = "objective" if example["label"] == "OBJ" else "subjective"

        out_prompt = (
            out_prompt
            + "Example "
            + str(index)
            + ":"
            + "\n"
            + "sentence: "
            + example["input"]
            + "\nlabel: "
            + label
            + "\n\n"
        )

    out_prompt = out_prompt + "sentence: " + input_sample + "\nlabel: \n"

    return out_prompt


def post_process(response):
    label = response["choices"][0]["message"]["content"].lower()

    if "label: objective" in label:
        label_fixed = "OBJ"
    elif "label: subjective" in label:
        label_fixed = "SUBJ"
    elif label == "objective" or label == "objective.":
        label_fixed = "OBJ"

    elif label == "subjective" or label == "subjective.":
        label_fixed = "SUBJ"

    return label_fixed
