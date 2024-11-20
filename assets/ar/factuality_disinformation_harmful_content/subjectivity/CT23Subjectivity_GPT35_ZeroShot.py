from llmebench.datasets import CT23SubjectivityDataset
from llmebench.models import LegacyOpenAIModel
from llmebench.tasks import SubjectivityTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-35-turbo (version 0301)",
        "description": "GPT35 model hosted on Azure, using the Completion API. API version '2023-03-15-preview'.",
        "scores": {"Macro-F1": "0.670"},
    }


def config():
    return {
        "dataset": CT23SubjectivityDataset,
        "task": SubjectivityTask,
        "model": LegacyOpenAIModel,
        "model_args": {
            "class_labels": ["SUBJ", "OBJ"],
            "max_tries": 3,
        },
        "general_args": {"test_split": "ar/dev"},
    }


def prompt(input_sample):
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": f"Classify the sentence as Subjective or Objective. Provide only label.\n\ntext: {input_sample}label: ",
            }
        ],
    }


def post_process(response):
    label = response["choices"][0]["text"]
    if label == "Objective" or label == "Objective.":
        label_fixed = "OBJ"
    elif label == "Subjective" or label == "Subjective.":
        label_fixed = "SUBJ"

    return label_fixed
