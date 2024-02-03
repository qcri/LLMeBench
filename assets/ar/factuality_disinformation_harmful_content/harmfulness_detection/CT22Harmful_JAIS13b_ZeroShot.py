from llmebench.datasets import CT22HarmfulDataset
from llmebench.models import FastChatModel
from llmebench.tasks import HarmfulDetectionTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Jais-13b-chat",
        "description": "Locally hosted Jais-13b-chat model using FastChat.",
    }


def config():
    return {
        "dataset": CT22HarmfulDataset,
        "task": HarmfulDetectionTask,
        "model": FastChatModel,
        "model_args": {
            "class_labels": ["0", "1"],
            "max_tries": 30,
        },
        "general_args": {"test_split": "ar"},
    }


def prompt(input_sample):
    base_prompt = (
        f'Given the following tweet, analyze its content and identify if it could potentially cause harm to an individual, society, or an organization. Annotate the tweet as "harmful" if it has the potential to cause harm, or "not_harmful" if it does not.\n\n'
        f"tweet: {input_sample}\n"
        f"label: \n"
    )
    return [
        {
            "role": "user",
            "content": base_prompt,
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"].lower()

    if "label: harmful" in label:
        label_fixed = "1"
    elif "label: not_harmful" in label:
        label_fixed = "0"
    elif "label: " in label:
        arr = label.split("label: ")
        label = arr[1].strip()

    elif label.startswith("harmful") or label.startswith("yes"):
        label_fixed = "1"

    elif (
        label.startswith("no")
        or label == "label: safe"
        or label == "not_harmful"
        or "not harmful" in label
    ):
        label_fixed = "0"
    else:
        label = label.replace(".", "").strip().lower()
        label = label.replace("label:", "").strip()
        label_fixed = label

    return label_fixed
