import os

from arabic_llm_benchmark.datasets import Q2QSimDataset
from arabic_llm_benchmark.models import GPTModel
from arabic_llm_benchmark.tasks import Q2QSimDetectionTask


def config():
    return {
        "dataset": Q2QSimDataset,
        "dataset_args": {},
        "task": Q2QSimDetectionTask,
        "task_args": {},
        "model": GPTModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": "gpt",
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/STS/nsurl-2019-task8/test.tsv",
        },
    }


def prompt(input_sample):
    q1,q2 = input_sample.split("\t")
    prompt = f"Are the two questions below semantically similar? The output should be exactly in form yes or no.\n\nQ1: {q1}\nQ2: {q2}\nlabel: "
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": prompt,
            }
        ],
    }


def post_process(response):
    input_label = response["choices"][0]["text"]
    input_label = input_label.replace(".", "").strip().lower()
    pred_label = ""

    if (
        "yes" in input_label
        or "label: 1" in input_label
    ):
        pred_label = "1"
    if (
        input_label == "no"
        or input_label.startswith("no,")
        or "label: 0" in input_label
        or "label: no" in input_label
        or "not semantically similar" in input_label
    ):
        pred_label = "0"

    if pred_label == "":
        pred_label = None

    return pred_label
