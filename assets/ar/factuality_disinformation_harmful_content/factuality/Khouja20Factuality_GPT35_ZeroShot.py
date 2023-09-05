import os

from llmebench.datasets import Khouja20FactualityDataset
from llmebench.models import GPTModel, RandomGPTModel
from llmebench.tasks import Khouja20FactualityTask


def config():
    return {
        "dataset": Khouja20FactualityDataset,
        "dataset_args": {},
        "task": Khouja20FactualityTask,
        "task_args": {},
        "model": GPTModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": "gpt",
            "class_labels": ["true", "false"],
            "max_tries": 30,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/factuality_stance_khouja/claim/test.csv"
        },
    }


def prompt(input_sample):
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": f"Detect the information in the sentence as correct or incorrect. Use label as true or false.\n\ntext: {input_sample} \nlabel: \n",
            }
        ],
    }


def post_process(response):
    label = response["choices"][0]["text"].lower().replace(".", "")
    if "label: true" in label or label == "true":
        label_fixed = "true"
    elif "label: false" in label or label == "false":
        label_fixed = "false"
    else:
        label_fixed = None

    return label_fixed
