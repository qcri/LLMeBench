import json

from llmebench.datasets import ThatiARDataset
from llmebench.models import VLLMModel
from llmebench.tasks import SubjectivityTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "LLama 3 8b",
        "description": "Deployed on the local server.",
        "scores": {},
    }


def config():
    return {
        "dataset": ThatiARDataset,
        "task": SubjectivityTask,
        "model": VLLMModel,
        "model_args": {
            "class_labels": ["SUBJ", "OBJ"],
            "max_tries": 30,
        },
    }


def prompt(input_sample):
    assistant_prompt = """
        I am an Arabic AI assistant specialized in classifying news article sentences into subjective or objective. 
        A subjective sentence expresses personal opinions, feelings, or beliefs, while an objective sentence presents facts, data, or unbiased information.
    """

    prompt = f"""
        Classify the following Arabic 'sentence' as subjective or objective. Provide only the label. 
        Provide your response in the following JSON format: {{"label": "your label"}}. 
        Please provide JSON output only. No additional text.

        sentence: {input_sample}
        """

    return [
        {
            "role": "assistant",
            "content": assistant_prompt,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


def post_process(response):
    data = response["choices"][0]["message"]["content"]
    response = json.loads(data)
    label = response["label"]
    if "label: objective" in label:
        label_fixed = "OBJ"
    elif "label: subjective" in label:
        label_fixed = "SUBJ"
    elif label == "objective" or label == "objective.":
        label_fixed = "OBJ"

    elif label == "subjective" or label == "subjective.":
        label_fixed = "SUBJ"

    return label_fixed
