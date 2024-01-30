from llmebench.datasets import CT23SubjectivityDataset
from llmebench.models import FastChatModel
from llmebench.tasks import SubjectivityTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "JAIS-13b",
        "description": "Locally hosted JAIS-13b-chat model using FastChat.",
        "scores": {"Macro-F1": ""},
    }


def config():
    return {
        "dataset": CT23SubjectivityDataset,
        "task": SubjectivityTask,
        "model": FastChatModel,
        "general_args": {"test_split": "ar/dev"},
    }


def prompt(input_sample):
    base_prompt = (
        f"صنف الجملة إلى لاموضوعية أو موضوعية.\n\n"
        f"التغريدة: {input_sample}\n"
        f"التصنيف: \n"
    )
    return [
        {
            "role": "user",
            "content": base_prompt,
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"].lower().replace(".", "")

    if "لاموضوعية" in label:
        label_fixed = "SUBJ"
    elif (
        label == "موضوعية" or label == "التصنيف: موضوعية" or "التصنيف: موضوعية" in label
    ):
        label_fixed = "OBJ"
    else:
        label_fixed = None

    return label_fixed
