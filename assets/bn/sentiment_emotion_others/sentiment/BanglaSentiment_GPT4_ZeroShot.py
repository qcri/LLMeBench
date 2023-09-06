import os

from llmebench.datasets import BanglaSentimentDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import SentimentTask


def config():
    return {
        "dataset": BanglaSentimentDataset,
        "dataset_args": {},
        "task": SentimentTask,
        "task_args": {},
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["Positive", "Negative", "Neutral"],
            "max_tries": 20,
        },
        "general_args": {
            "data_path": "data/sentiment_emotion_others/sentiment/bn/bn_all_test.tsv",
        },
    }


def prompt(input_sample):
    prompt_string = f"""Based on the content of the text, please classify it as either "Positive", "Negative", or "Neutral". Provide only the label as your response. 

        text: {input_sample}

        label: """

    return [
        {
            "role": "system",
            "content": "You are a expert annotator. Your task is to analyze the text and identify sentiment polarity.",
        },
        {
            "role": "user",
            "content": prompt_string,
        },
    ]


def post_process(response):
    if not response:
        return None
    label = response["choices"][0]["message"]["content"]

    label_fixed = label.replace("label:", "").strip()
    if label_fixed.startswith("Please provide the text"):
        label_fixed = None

    return label_fixed
