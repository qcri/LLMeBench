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
            "engine_name": "gpt-4-32k",
            "class_labels": ["SUBJ", "OBJ"],
            "max_tries": 30,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/subjectivity/dev_ar.tsv"
        },
    }


def prompt(input_sample):
    prompt_string = (
        f'Annotate the "sentence" as subjective or objective\n\n'
        f"tweet: {input_sample}\n"
        f"label: \n"
    )
    return [
        {
            "role": "system",
            "content": "You are an expert annotator, who can analyze the information in the sentence and determine whether the sentence is subjective or objective.",  # You are capable of identifying and annotating tweets correct or incorrect
        },
        {
            "role": "user",
            "content": prompt_string,
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
    else:
        label_fixed = None

    return label_fixed
