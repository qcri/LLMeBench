from llmebench.datasets import CT23SubjectivityDataset
from llmebench.models import FastChatModel
from llmebench.tasks import SubjectivityTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Jais-13b-chat",
        "description": "Locally hosted Jais-13b-chat model using FastChat.",
        "scores": {"Macro-F1": "0.5717"},
    }


def config():
    return {
        "dataset": CT23SubjectivityDataset,
        "task": SubjectivityTask,
        "model": FastChatModel,
        "general_args": {"test_split": "ar/dev"},
    }


def prompt(input_sample):
    base_prompt = f"Classify the sentence as Subjective or Objective. Provide only label.\n\ntext: {input_sample}label: "
    return [
        {
            "role": "user",
            "content": base_prompt,
        },
    ]


def post_process(response):
    label = (
        response["choices"][0]["message"]["content"].lower().replace(".", "").strip()
    )

    if "label: objective" in label:
        label_fixed = "OBJ"
    elif "label: subjective" in label:
        label_fixed = "SUBJ"
    elif label == "objective" or label == "objective.":
        label_fixed = "OBJ"

    elif label == "subjective" or label == "subjective.":
        label_fixed = "SUBJ"

    return label_fixed
