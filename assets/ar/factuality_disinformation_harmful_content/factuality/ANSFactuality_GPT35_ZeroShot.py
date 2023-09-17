from llmebench.datasets import ANSFactualityDataset
from llmebench.models import LegacyOpenAIModel
from llmebench.tasks import FactualityTask


def config():
    return {
        "dataset": ANSFactualityDataset,
        "dataset_args": {},
        "task": FactualityTask,
        "task_args": {},
        "model": LegacyOpenAIModel,
        "model_args": {
            "class_labels": ["true", "false"],
            "max_tries": 30,
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