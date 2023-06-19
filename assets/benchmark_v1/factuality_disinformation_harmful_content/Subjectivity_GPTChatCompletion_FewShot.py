import os
import random
import re

from arabic_llm_benchmark.datasets import SubjectivityDataset
from arabic_llm_benchmark.models import GPTChatCompletionModel
from arabic_llm_benchmark.tasks import SubjectivityTask


random.seed(1333)


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
                "train_data_path": "data/factuality_disinformation_harmful_content/subjectivity/train_ar.tsv",
            },
        },
    }


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n\n"
    for example in examples:
        out_prompt = (
            out_prompt
            + "tweet: "
            + example["input"]
            + "\nlabel: "
            + example["label"]
            + "\n\n"
        )

    out_prompt = out_prompt + "tweet: " + input_sample + "\nlabel: \n"

    return out_prompt


def prompt(input_sample, examples):
    base_prompt = f'Annotate the "sentence" as subjective or objective'
    return [
        {
            "role": "system",
            "content": "You are an expert annotator, who can analyze the information in the sentence and determine whether the sentence is subjective or objective.",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


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
    elif "label: obj" in label or label == "obj" or label == "obj.":
        label_fixed = "OBJ"
    elif "label: subj" in label or label == "subj" or label == "subj.":
        label_fixed = "SUBJ"
    else:
        label_fixed = None

    return label_fixed
