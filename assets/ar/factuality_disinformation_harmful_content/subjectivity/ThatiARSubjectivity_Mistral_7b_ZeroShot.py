import json

from llmebench.datasets import ThatiARDataset
from llmebench.models import AzureModel
from llmebench.tasks import SubjectivityTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "LLama 3 8b",
        "description": "Deployed on Azure.",
        "scores": {},
    }


def config():
    return {
        "dataset": ThatiARDataset,
        "task": SubjectivityTask,
        "model": AzureModel,
        "model_args": {
            "class_labels": ["SUBJ", "OBJ"],
            "max_tries": 30,
        },
    }


def prompt(input_sample):

    assistant_prompt = """
        I am an Arabic AI assistant specialized in classifying sentences into subjective or objective.         
    """

    prompt = f"""
        Classify the following Arabic 'sentence' as subjective or objective. Provide only the label. Please do not provide any additional text. 

        sentence: {input_sample}

        label: 
        """

    return [
        {
            "role": "user",
            "content": prompt,
        },
        {
            "role": "assistant",
            "content": assistant_prompt,
        },
    ]


def post_process(response):
    label = response["output"]

    if "label: objective" in label or "objective" in label or "Objective" in label:
        label_fixed = "OBJ"
    elif "label: subjective" in label or "subjective" in label or "Subjective" in label:
        label_fixed = "SUBJ"
    elif label == "objective" or label == "objective.":
        label_fixed = "OBJ"

    elif label == "subjective" or label == "subjective.":
        label_fixed = "SUBJ"
    else:
        label_fixed = None

    return label_fixed
