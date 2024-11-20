import json

from llmebench.datasets import ThatiARDataset
from llmebench.models import AnthropicModel
from llmebench.tasks import SubjectivityTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "claude-3-5-sonnet-20240620",
        "description": "Anthropic model - claude-3-5-sonnet. Find more https://www.anthropic.com/news/claude-3-5-sonnet",
        "scores": {},
    }


def config():
    system_msg = "AI assistant specialized in classifying news article sentences into subjective or objective. A subjective sentence expresses personal opinions, feelings, or beliefs, while an objective sentence presents facts, data, or unbiased information."
    return {
        "dataset": ThatiARDataset,
        "task": SubjectivityTask,
        "model": AnthropicModel,
        "model_args": {
            "class_labels": ["SUBJ", "OBJ"],
            "max_tries": 30,
            "system": system_msg,
        },
    }


def prompt(input_sample):

    prompt = f"""
        Classify the following Arabic 'sentence' as subjective or objective. Provide only the label. 
        Provide your response in the following JSON format: {{"label": "your label"}}. 
        Please provide JSON output only. No additional text.

        sentence: {input_sample}
        """
    return [
        {
            "role": "user",
            "content": prompt,
        },
    ]


def post_process(response):
    data = response["content"][0]["text"].lower()
    data = json.loads(data)
    label = data["label"]
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
